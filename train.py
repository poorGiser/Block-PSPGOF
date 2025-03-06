#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import diptest
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal

def get_view2gaussian(rotation,viewmatrix,xyz):
    r = rotation
    N = xyz.shape[0]
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]
    
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    rots = R
    # N = xyz.shape[0] 
    G2W = torch.zeros((N, 4, 4), device='cuda')
    G2W[:, :3, :3] = rots # TODO check if we need to transpose here
    G2W[:, :3, 3] = xyz
    G2W[:, 3, 3] = 1.0
    
    viewmatrix = viewmatrix.transpose(0, 1)
    G2V = viewmatrix @ G2W#高斯坐标向camera坐标转换
    
    R = G2V[:, :3, :3]
    t = G2V[:, :3, 3]
    
    #相当于求逆
    t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
    V2G = torch.zeros((N, 4, 4), device='cuda')
    V2G[:, :3, :3] = R.transpose(1, 2)
    V2G[:, :3, 3] = t2
    V2G[:, 3, 3] = 1.0
    
    # transpose view2gaussian to match glm in CUDA code
    # V2G = V2G.transpose(2, 1).contiguous()
    return V2G

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image
    
def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad

def get_edge_aware_distortion_map_from_depth(depth_image, distortion_map):
    depth_image = depth_image.unsqueeze(0)
    grad_img_left = torch.mean(torch.abs(depth_image[:, 1:-1, 1:-1] - depth_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(depth_image[:, 1:-1, 1:-1] - depth_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(depth_image[:, 1:-1, 1:-1] - depth_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(depth_image[:, 1:-1, 1:-1] - depth_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
    

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    #TODO：输出图片
    # output_image = image.detach().cpu().squeeze().permute(1,2,0).numpy()* 255
    # output_trans_image = transformed_image.detach().cpu().squeeze().permute(1,2,0).numpy() * 255
    # cv2.imwrite(f'/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble7/block0/log_transformer_image/{str(view_idx)}_transformed.jpg', cv2.cvtColor(output_trans_image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(f'/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble7/block0/log_transformer_image/{str(view_idx)}.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def calc_diff(mode_img, depth_img):
    mean = torch.mean(depth_img)
    std = torch.std(depth_img)
    diff = (depth_img - mode_img) / (depth_img + max(depth_img.min(), mean - 3*std))
    return diff
def calc_alpha(means2D, conic_opac, x, y):
    dx = x - means2D[:,0]
    dy = y - means2D[:,1]
    power = -0.5*(conic_opac[:,0]*(dx*dx) + conic_opac[:,2]*(dy*dy)) - conic_opac[:,1]*dx*dy
    alpha = power
    alpha[power > 0] = -100
    return alpha
def prune_floaters(viewpoint_stack, gaussians, pipe, background, dataset, iteration):
     with torch.no_grad():
        N = gaussians.get_opacity.shape[0]
        ctrs = [0]*len(viewpoint_stack)

        num_pixels_init = [None]*len(viewpoint_stack)
        #mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"depth_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"diff_{iteration}"), exist_ok=True)
        
        temp_info_path = os.path.join(dataset.model_path, f"temp_{iteration}")
        os.makedirs(temp_info_path, exist_ok=True)


        mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

        plt.figure(figsize=(25,20))
        
        
        #阻止OOM 保存到本地
        dips = []        
        # point_lists = []
        # means2Ds = []
        # conic_opacities = []
        # mode_ids = []
        # diffs = []
        names = []
        
        for view in tqdm(viewpoint_stack):
            one_info = {}
            names.append(view.image_name)
            render_pkg = render(view, gaussians, pipe, background, kernel_size=dataset.kernel_size,rt_depths=True)
            mode_id, mode, point_list, depth, means2D, conic_opacity = render_pkg["mode_id"], render_pkg["modes"], render_pkg["point_list"], render_pkg["alpha_depth"], render_pkg["means2D"], render_pkg["conic_opacity"] 
            diff = calc_diff(mode, depth)
            plt.imsave(os.path.join(dataset.model_path, f"modes_{iteration}", f"{view.image_name}.png" ), mode.cpu().numpy().squeeze(), cmap='jet')
            plt.imsave(os.path.join(dataset.model_path, f"depth_{iteration}", f"{view.image_name}.png" ), depth.cpu().numpy().squeeze(), cmap='jet')
            dip_value = diptest.dipstat(diff[diff > 0].cpu().numpy())
            
            # point_lists.append(point_list)
            # means2Ds.append(means2D)
            # conic_opacities.append(conic_opacity)
            # mode_ids.append(mode_id)
            # diffs.append(diff)
            one_info["point_list"] = point_list
            one_info["means2D"] = means2D
            one_info["conic_opacity"] = conic_opacity
            one_info["mode_id"] = mode_id
            one_info["diff"] = diff
            one_info["dip"] = dip_value
            
            #save camera params
            one_info["fx"] = view.focal_x
            one_info["fy"] = view.focal_y
            one_info["height"] = view.image_height
            one_info["width"] = view.image_width
            one_info["world_view_transform"] = view.world_view_transform#transposed
            
            torch.save(one_info,os.path.join(temp_info_path,view.image_name+".pt"))
            dips.append(dip_value)

        dips = np.array(dips)
        avg_dip = dips.mean()
        perc = dataset.prune_perc*100*np.exp(-1*dataset.prune_exp*avg_dip)

        if (perc < 80):
            perc = 80
        print(f'Percentile {perc}')
        
        rotation = gaussians.get_rotation
        xyz = gaussians.get_xyz
        scale = gaussians.get_scaling_with_3D_filter
        for name in names:
            #TODO:读取文件获取信息
            info_path = os.path.join(temp_info_path,name+".pt")
            info = torch.load(info_path)
            mode_id = info["mode_id"]
            point_list = info["point_list"]
            diff = info["diff"]
            means2D = info["means2D"]
            conic_opacity = info["conic_opacity"]
            W = info["width"]
            H = info["height"]
            focal_x = info["fx"]
            focal_y = info["fy"]
            world_view_transform = info["world_view_transform"]#transposed
            
            submask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
           
            diffpos = diff[diff > 0]
            threshold = np.percentile(diffpos.cpu().numpy(), perc)
            pruned_modes_mask = (diff > threshold).squeeze()
            cv2.imwrite(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}",f"{name}.png"), pruned_modes_mask.cpu().numpy().squeeze().astype(np.uint8)*255)

            pixel_y, pixel_x = torch.meshgrid(torch.arange(pruned_modes_mask.shape[0]), torch.arange(pruned_modes_mask.shape[1]), indexing='ij')
            pixel_y = pixel_y.to('cuda')
            pixel_x = pixel_x.to('cuda')
            prune_mode_ids = mode_id[:,pruned_modes_mask] # subselect the mode idxs
            pixel_x = pixel_x[pruned_modes_mask]
            pixel_y = pixel_y[pruned_modes_mask]

            neg_mask = (prune_mode_ids == -1).any(dim=0)
            prune_mode_ids = prune_mode_ids[:,~neg_mask]
            pixel_x = pixel_x[~neg_mask]
            pixel_y = pixel_y[~neg_mask]

            selected_gaussians = set()
            NEAR_PLANE = 0.2
                        
            for j in tqdm(range(prune_mode_ids.shape[-1])):
                x = pixel_x[j]#1
                y = pixel_y[j]#1
                gausses = point_list[prune_mode_ids[0,j]:prune_mode_ids[1,j]+1].long()
                
                #for 3d gaussian alpha
                # c_opacs = conic_opacity[gausses]
                # m2Ds = means2D[gausses]
                # test_alpha = calc_alpha(m2Ds, c_opacs, x, y)
                
                #for gof alpha 
                ray = ((x - W/2.) / focal_x, (y - H/2.) / focal_y)
                ray_point =torch.tensor([ray[0] , ray[1], 1.0]).reshape(-1,1)
                
                gaussian_rotation = rotation[gausses].cpu()
                gaussian_xyz = xyz[gausses].cpu()
                view2gaussian_j = get_view2gaussian(gaussian_rotation,world_view_transform,gaussian_xyz).cpu()
                scale_j = scale[gausses].cpu()
                
                cam_pos_local = view2gaussian_j[:,:3,3]#高斯坐标系下的cam_pos
                ray_local = (view2gaussian_j[:,:3,:3]@ray_point).squeeze(-1)
                ray_local_scaled = ray_local / scale_j
                cam_pos_local_scaled = cam_pos_local / scale_j
                AA = torch.sum(torch.pow(ray_local_scaled,2),dim=-1)
                BB = 2 * torch.sum(ray_local_scaled * cam_pos_local_scaled,dim=-1)
                CC = torch.sum(torch.pow(cam_pos_local_scaled,2),dim=-1)
                t = -BB/(2*AA)
                # if (t <= NEAR_PLANE):
				#     continue
                min_value = -(BB/AA) * (BB/4.) + CC
                power = -0.5 * min_value
                alpha = power
                alpha[power > 0] = -100
                
                alpha_mask = (alpha > dataset.power_thresh) * (t >= NEAR_PLANE)
                gausses = gausses[alpha_mask]
       
                selected_gaussians.update(gausses.tolist())
      
            submask[list(selected_gaussians)] = True
            
            # print(f"submask {torch.count_nonzero(submask)}")

            mask = mask | submask

            # num_points_pruned = submask.sum()
            # print(f'Pruning {num_points_pruned} gaussians')

        print(gaussians.get_xyz.shape[0])
        gaussians.prune_points(mask)
        print(gaussians.get_xyz.shape[0])
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,prune_sched):
    # print(prune_sched)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.setPlaneIso(opt.plane_iso)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        #将图片加载进cuda
        viewpoint_cam.load_in_cuda()
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering[:3, :, :]
        
        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # depth distortion regularization
        depth = rendering[6, :, :]
        distortion_map = rendering[8, :, :]
        
        # distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)#在深度变化小的地方(梯度大)高斯要更加紧凑
        # distortion_map = get_edge_aware_distortion_map_from_depth(depth, distortion_map)#在深度变化小的地方(梯度大)高斯要更加紧凑
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)#3 H W

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()
        
        lambda_distortion = opt.lambda_distortion if iteration >= opt.distortion_from_iter else 0.0
        lambda_depth_normal = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0
        
        #plane loss
        if iteration > opt.plane_from_iter:
            plane_loss = opt.lambda_plane * torch.mean(torch.abs(torch.gather(gaussians.get_scaling[gaussians.plane_mask],dim=1,index = gaussians.smallest_dir)))
            loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion + plane_loss
        # Final loss
        else:
            loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion
        
        
        loss.backward()
        
        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % opt.densification_interval == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                rendering = render(eval_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)["render"]
                image = rendering[:3, :, :]
                transformed_image = L1_loss_appearance(image, eval_cam.original_image.cuda(), gaussians, eval_cam.idx, return_transformed_image=True)
                
                normal = rendering[3:6, :, :]
                normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                
            # transform to world space
            c2w = (eval_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = (normal + 1.) / 2.
            
            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
            depth_normal = (depth_normal + 1.) / 2.
            depth_normal = depth_normal.permute(2, 0, 1)
            
            gt_image = eval_cam.original_image.cuda()
            
            depth_map = apply_depth_colormap(depth[..., None], rendering[7, :, :, None], near_plane=None, far_plane=None)
            depth_map = depth_map.permute(2, 0, 1)
            
            accumlated_alpha = rendering[7, :, :, None]
            colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
            colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
            
            distortion_map = rendering[8, :, :]
            distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)
        
            row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
            row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, transformed_image], dim=2)
            
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iteration}.jpg")
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    #TODO：trim gs：scale driven trim
                    # scene_mask:能筛选出相机包围区域的mask
                    # scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                    # gaussians.densify_and_scale_split(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, opt.densify_scale_factor, scene_mask=None, N=2, no_grad=True)
                    # prune_mask = (gaussians.get_opacity < 0.05).squeeze()
                    # if size_threshold:
                    #     big_points_vs = gaussians.max_radii2D > size_threshold
                    #     big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                    #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                    # gaussians.prune_points(prune_mask)
                    
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    
                    if iteration > opt.plane_from_iter:
                        gaussians.updatePlaneMask()

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
            if iteration % opt.plane_interval == 0:
                gaussians.updatePlaneMask()
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if iteration % opt.plane_interval == 0:
                bl = torch.sum(gaussians.plane_mask)/gaussians.get_xyz.shape[0]
                with open(args.model_path+'/plane_occ.txt', 'a') as file:
                    # 写入内容到文件末尾
                    file.write(str(bl.item()) + "\n")
            if iteration in prune_sched:
                pass
                # os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
                # os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
                # scene.save(iteration-1)
                # prune_floaters(scene.getTrainCameras().copy(), gaussians, pipe, background, dataset, iteration)
                # gaussians.compute_3D_filter(cameras=trainCameras)
                # scene.save(iteration+1)
                # last_prune_iter = iteration
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        viewpoint_cam.load_in_cpu()
        
            
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("-block_index", type=int, default = 0)
    parser.add_argument("-resolution", type=int, default = 4)
    parser.add_argument("-con", type=bool, default = True)
    parser.add_argument("-gpu_count", type=int, default = 0)
    
    #Ablation Experiments params
    parser.add_argument("-no_block", type=bool, default = False) 
    parser.add_argument("-block_num", type=int, default = 8) 
    
    #floater prune iter
    # parser.add_argument("--prune_sched", nargs="+", type=int, default=[10000,20000])
    parser.add_argument("--prune_sched", nargs="+", type=int, default=[5000,10000])
    # parser.add_argument("--prune_sched", nargs="+", type=int, default=[10000])
    
    args = parser.parse_args(sys.argv[1:])
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.set_device(torch.device(f"cuda:{args.gpu_count}"))
    # print(torch.device(f"cuda:{args.gpu_count}"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.prune_sched)

    # All done
    print("\nTraining complete.")
