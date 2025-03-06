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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor,filter_gaussian,index):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")\

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # gaussians.to_cpu()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if filter_gaussian:
            if idx == index:
            #只筛选落在视锥体内部的gaussian
            # if idx != 0:
            #     del gaussians
            #     torch.cuda.empty_cache()
            #     gaussians = GaussianModel(3)
            #     gaussians.load_ply(os.path.join(model_path,
            #                                         "point_cloud",
            #                                         "iteration_" + str(iteration),
            #                                         "point_cloud.ply"))
                raw_xyz = gaussians.get_xyz.transpose(0, 1)#4 * n
                w2c = view.world_view_transform.transpose(0, 1)#4 * 4
                w2n = view.full_proj_transform.transpose(0, 1)
                raw_xyz_q = torch.cat([raw_xyz, torch.ones_like(raw_xyz[:1, ...])], dim=0)
                xyz_camera = w2c @ raw_xyz_q
                xyz_proj = w2n @ raw_xyz_q
                
                # ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
                iso = 5.0
                filter_mask = (xyz_camera[2,...] > 0.2) & (xyz_proj[0,...] > -iso) & (xyz_proj[0,...] < iso) & (xyz_proj[1,...] > -iso) & (xyz_proj[1,...] < iso)
                
                gaussians.filter_mask(filter_mask)   
                rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size,mask=filter_mask)["render"]
                rendering = rendering[:3, :, :]
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        else:
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size,mask=None)["render"]
            rendering = rendering[:3, :, :]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,filter_gaussian:bool,index:int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,onlyTest=True)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor,filter_gaussian = filter_gaussian,index = index)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor,filter_gaussian = filter_gaussian,index = index)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("-block_index", type=int, default = 0)
    parser.add_argument("-resolution", type=int, default = 4)
    parser.add_argument("-con", type=bool, default = True)
    
    parser.add_argument("-no_block", type=bool, default = False) 
    
    #是否对gaussian进行过滤,防止oom
    parser.add_argument("-filter_gaussian", type=bool, default = False)
    parser.add_argument("-index", type=int, default = 0)
    
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.filter_gaussian,args.index)