'''
渲染视频
'''
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
import mediapy as media
from tqdm import tqdm
import numpy as np
import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2
import math
from scipy.spatial.transform import Rotation
def rotation_matrix_to_quaternion(R):
    # 将旋转矩阵转换为四元数
    r = Rotation.from_matrix(R)
    return r.as_quat()  # 返回四元数 [x, y, z, w]

def quaternion_to_rotation_matrix(q):
    # 将四元数转换为旋转矩阵
    r = Rotation.from_quat(q)
    return r.as_matrix()  # 返回旋转矩阵
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"video_imgs_{scale_factor}")

    makedirs(render_path, exist_ok=True)
    
    n_fames = 60
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        rendering = rendering[:3, :, :]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
    images = [img for img in os.listdir(render_path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(render_path, images[0]))
    height, width, layers = frame.shape

    # 指定输出视频的编解码器
    video_name = os.path.join(os.path.join(model_path, name, "ours_{}".format(iteration),"fly.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, n_fames, (width, height))  # 这里的 1 表示帧率为 1，可以根据实际需求调整
    
    def sort_by_number(path):
        return int(path.split(".")[0])
    images = sorted(images, key=sort_by_number)
    # 写入图片到视频
    for image in images:
        video.write(cv2.imread(os.path.join(render_path, image)))
    cv2.destroyAllWindows()
    video.release()
    print(f'视频 {video_name} 已生成')
    
def slerp(q1, q2, t):
    # 四元数的球面线性插值
    dot_product = np.dot(q1, q2)

    # 如果点积小于0，反向四元数以确保最短路径
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # 夹角的余弦值
    if dot_product > 0.9995:
        # 如果夹角很小，使用线性插值
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # 计算插值
    theta_0 = np.arccos(dot_product)  # 夹角
    theta = theta_0 * t  # 插值夹角

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return (s1 * q1 + s2 * q2)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,onlyTest=True)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        #插值位姿
        # need_cameras = 10
        interval = 100
        cameras = scene.getTestCameras()
        cameras = cameras[:10]
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in cameras])
        cameras_num = len(c2ws)
        
        #计算相机距离的平均值
        distance = []
        for i in range(cameras_num - 1):
            start_pos = c2ws[i][:3,3]
            end_pos = c2ws[i + 1][:3,3]
            distance.append(np.linalg.norm(start_pos - end_pos))
        mean_distance = 1
        
        allCamera = []
        for i in range(cameras_num - 1):
            start_camera = cameras[i]
            old_R = c2ws[i][:3,:3]
            
            last_camera = start_camera
            
            start_position = c2ws[i][:3,3]
            end_position = c2ws[i + 1][:3,3]
            dis = np.linalg.norm(start_position - end_position)
            
            one_interval = math.floor(interval * (dis / mean_distance))
            span = (end_position - start_position) / one_interval
            
            # end_camera = cameras[i + 1]
            
            start_rot = c2ws[i][:3,:3]
            end_rot = c2ws[i+1][:3,:3]
            
            start_q = rotation_matrix_to_quaternion(start_rot)
            end_q = rotation_matrix_to_quaternion(end_rot)
            for j in range(one_interval):
                bl = max(j / one_interval,1)
                new_camera = copy.deepcopy(last_camera)
                #插值生成新相机位置
                q_interp = slerp(start_q, end_q, bl)
                new_pos = start_position + span * j
                
                R_interp = quaternion_to_rotation_matrix(q_interp)
                
                #修改new_camera参数
                # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
                # new_camera.world_view_transform = torch.tensor(getWorld2View2(old_R, new_pos, new_camera.trans, new_camera.scale)).transpose(0, 1).cuda()
                new_c2w = np.concatenate((R_interp,new_pos.reshape(3,1)),axis=1)
                new_c2w = np.concatenate((new_c2w,np.asarray([[0,0,0,1]])),axis=0).astype(np.float32)
                new_camera.world_view_transform = torch.tensor(np.linalg.inv(new_c2w)).transpose(0, 1).cuda()
                new_camera.full_proj_transform = (new_camera.world_view_transform.unsqueeze(0).bmm(new_camera.projection_matrix.unsqueeze(0))).squeeze(0)
                new_camera.camera_center = new_camera.world_view_transform.inverse()[3, :3]
                
                allCamera.append(new_camera)
                
                last_camera = new_camera
        
        # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)
        render_set(dataset.model_path, "train", scene.loaded_iter, allCamera, gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("-block_index", type=int, default = 0)
    parser.add_argument("-resolution", type=int, default = 4)
    parser.add_argument("-con", type=bool, default = True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))