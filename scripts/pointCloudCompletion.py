'''
基于密度的点云补全算法
'''
import open3d as o3d
import os 
import numpy as np
import json
import math
from tqdm import tqdm
from scipy.spatial import KDTree
from tqdm import tqdm

def nearest_neighbor_interpolation(points, colors, query_points):
    tree = KDTree(points)
    _, indices = tree.query(query_points)
    return colors[indices]

def pointCloudCompletion(root_dir,block_index,plane_block_num):
    pcd_cov_path = os.path.join(root_dir,"blocks_manh",str(block_index),f"pcd_{block_index}_cov.ply")
    pcd = o3d.io.read_point_cloud(pcd_cov_path)
    origin_pcd = np.asarray(pcd.points).copy()
    points,_ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.0)
    points = np.asarray(pcd.points)
    points_colors = np.asarray(pcd.colors)
    block_infos_path = os.path.join(root_dir,"blocks_manh","blocks_info.json")
    with open(block_infos_path) as file:
        block_infos = json.load(file)
    for i in range(len(block_infos)):
        if block_infos[i]['block_id'] == block_index:
            block_info = block_infos[i]
    
    area = block_info["orgin_block_range"]
    # area = [np.min(points[...,0]),np.min(points[...,2]),np.max(points[...,0]),np.max(points[...,1])]
    
    #筛选在其中的点云
    block_point_mask = (points[...,0] > area[0]) * (points[...,0] < area[2]) * (points[...,2] > area[1]) * (points[...,2] < area[3])
    block_points = points[block_point_mask]
    block_points_colors = points_colors[block_point_mask]
    
    block_pcd = o3d.geometry.PointCloud()
    block_pcd.points = o3d.utility.Vector3dVector(block_points)
    block_pcd.colors = o3d.utility.Vector3dVector(block_points_colors)
    block_pcd,index= block_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    block_points = np.asarray(block_pcd.points)
    block_points_colors = np.asarray(block_pcd.colors)
    block_points_num = block_points.shape[0]
    #TODO:1
    
    area = [np.min(block_points[...,0]),np.min(block_points[...,2]),np.max(block_points[...,0]),np.max(block_points[...,2])]
    
    #地面区域检测
    height = np.max(block_points[...,1]) - np.min(block_points[...,1])
    max_height = np.max(block_points[...,1])
    min_height = np.min(block_points[...,1])
    total_fenduan = 10
    ground_height = height / total_fenduan
    init_height = max_height
    height_add = 0.01
    max_point_num = 0
    ground_points_iso = block_points_num / 10
    
    while(init_height > min_height):
        point_num = np.sum((block_points[...,1] < init_height) * (block_points[...,1] > (init_height - ground_height)))
        # print(point_num)
        if point_num > max_point_num:
            # ground = [init_height,init_height + ground_height]
            ground = [init_height - ground_height,max_height]
            
            max_point_num = point_num
        init_height -= height_add
        
        # if point_num > ground_points_iso:
        #     # ground = [init_height,init_height + ground_height]
        #     # ground = [init_height - ground_height,init_height]
        #     ground = [init_height - ground_height,max_height]
        #     break
    
    ground_points = block_points[(block_points[...,1] > ground[0]) * (block_points[...,1] < ground[1])]
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(ground_points)
    # o3d.io.write_point_cloud("ground_points.ply", pc)
    ground_points_num = ground_points.shape[0]
    
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(block_points)
    # o3d.io.write_point_cloud("origin.ply", pc)
    #计算密度
    block_max_len = (area[2] - area[0]) if (area[2] - area[0]) > (area[3] - area[1]) else (area[3] - area[1])
    width = block_max_len / plane_block_num
    
    x_block_num = int((area[2] - area[0]) // width)
    z_block_num = int((area[3] - area[1]) // width)
        
    # density = block_points_num / (x_block_num * z_block_num)
    density = ground_points_num / (x_block_num * z_block_num)
    new_points = []
    new_points_colors = []
    for i in tqdm(range(x_block_num)):
        for j in range(z_block_num):
            block_x_min = area[0] + i * width
            block_x_max = area[0] + (i * width) + width
            block_z_min = area[1] + j * width
            block_z_max = area[1] + j * width + width
            
            if i == (x_block_num - 1):
                block_x_max = area[2]
            if j == (z_block_num - 1):
                block_z_max = area[3]
            point_mask = (block_points[...,0] > block_x_min) * (block_points[...,0] < block_x_max) * (block_points[...,2] > block_z_min) * (block_points[...,2] < block_z_max)
            ground_point_mask = (block_points[...,0] > block_x_min) * (block_points[...,0] < block_x_max) * (block_points[...,2] > block_z_min) * (block_points[...,2] < block_z_max) * (block_points[...,1] > ground[0]) * (block_points[...,1] < ground[1])
            one_block_num = np.sum(point_mask)
            one_block_points = block_points[point_mask]
            
            one_block_colros = block_points_colors[point_mask]
            
            one_block_ground_points = block_points[ground_point_mask]
            one_block_ground_num = block_points[ground_point_mask].shape[0]
            
            # print(one_block_ground_num)
            if one_block_ground_num < density:
                add_point_num = math.ceil(density - one_block_ground_num)
                # print(one_block_num,one_block_ground_num)
                if (one_block_num - one_block_ground_num) > density:#非地面区域
                    new_points.append(one_block_points)
                    new_points_colors.append(one_block_colros)
                    continue
                if one_block_ground_num <= 1:
                    #随机采样
                    x = np.random.uniform(block_x_min, block_x_max, add_point_num)
                    y = np.random.uniform(ground[0], ground[1], add_point_num)
                    z = np.random.uniform(block_z_min, block_z_max, add_point_num)
                    
                    add_points = np.stack([x,y,z],axis=-1)
                    one_all_points = np.concatenate([one_block_points,add_points],axis=0)
                    
                    # new_points_colors.append(np.ones_like((one_all_points)) * 0.5) 
                    new_points_color = nearest_neighbor_interpolation(block_points,block_points_colors , one_all_points)
                else:
                    #高斯采样 
                    one_block_mean = np.mean(one_block_ground_points, axis=0)
                    one_block_cov = np.cov(one_block_ground_points,rowvar=False)
                    add_points = np.random.multivariate_normal(one_block_mean, one_block_cov, add_point_num)
                    one_all_points = np.concatenate([one_block_points,add_points],axis=0)
                    new_points_color = nearest_neighbor_interpolation(one_block_points, one_block_colros, one_all_points)
                #最近邻插值生成颜色
                # new_points_color = nearest_neighbor_interpolation(one_block_points, one_block_colros, one_all_points)
                new_points.append(one_all_points)
                new_points_colors.append(new_points_color) 
            else:
                new_points.append(one_block_points)
                new_points_colors.append(one_block_colros)
                
    new_points = np.concatenate(new_points,axis=0)
    new_points_colors = np.concatenate(new_points_colors,axis=0)
    
    
    #和区域之外的合并
    print(block_points.shape[0])
    print(new_points.shape[0])
    
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(new_points)
    # o3d.io.write_point_cloud("add.ply", pc)
    
    block_point_mask_new = (origin_pcd[...,0] > area[0]) * (origin_pcd[...,0] < area[2]) * (origin_pcd[...,2] > area[1]) * (origin_pcd[...,2] < area[3])
    block_out_mask = ~block_point_mask_new
    block_out_points = points[block_out_mask]
    block_out_points_colors = points_colors[block_out_mask]
    
    all_points = np.concatenate([block_out_points,new_points],axis=0)
    all_points_colros =  np.concatenate([block_out_points_colors,new_points_colors],axis=0)
    all_points_normals =  np.zeros_like(all_points_colros)
    
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)
    pc.normals = o3d.utility.Vector3dVector(all_points_normals)
    pc.colors = o3d.utility.Vector3dVector(all_points_colros)
    
    o3d.io.write_point_cloud(os.path.join(root_dir,"blocks_manh",str(block_index),f"pcd_{block_index}_cov_fill.ply"), pc)
    
    # print(points.shape[0])
    # print(all_points.shape[0])
    
    print("Done!")

if __name__ == "__main__":
    root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js"
    
    # plane_block_num = 128
    # for i in tqdm(range(1)): 
    #     pointCloudCompletion(root_dir=root_dir,block_index=i,plane_block_num=plane_block_num)
        
    # plane_block_num = 128
    # for i in tqdm(range(4)): 
    #     pointCloudCompletion(root_dir=root_dir,block_index=i,plane_block_num=plane_block_num)
    
    # plane_block_num = 128
    # for i in tqdm(range(8)): 
    #     pointCloudCompletion(root_dir=root_dir,block_index=i,plane_block_num=plane_block_num)
    
    # plane_block_num = 128
    # for i in tqdm(range(8)): 
    #     pointCloudCompletion(root_dir=root_dir,block_index=i,plane_block_num=plane_block_num)
    
    plane_block_num = 128
    for i in tqdm(range(4)): 
        pointCloudCompletion(root_dir=root_dir,block_index=i,plane_block_num=plane_block_num)
    