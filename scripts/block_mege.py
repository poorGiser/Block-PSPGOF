'''
merge all optimized blocks pcd
将所有优化好的pcd合并在一起
'''
from argparse import ArgumentParser, Namespace
import os
import sys
import json
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from scene import GaussianModel

def remove_duplicates(data, key):
    seen = set()
    result = []
    for item in data:
        value = item[key]
        if value not in seen:
            seen.add(value)
            result.append(item)
    return result

def block_mege(data_path,blocks_path,optimizer_pcds_path,save_path,iter_nums=[30000]):
    #1 cfg_args file
    parser = ArgumentParser(description="block_mege")
    parser.add_argument("-sh_degree", type=int, default = 3)
    parser.add_argument("-resolution", type=int, default = 4)
    parser.add_argument("-white_background", type=bool, default = False)
    parser.add_argument("-data_device", type=str, default = "cuda")
    parser.add_argument("-eval", type=bool, default = False)
    
    parser.add_argument("-images", type=str, default = "images")
    parser.add_argument("-con", type=bool, default = True)
    parser.add_argument("-kernel_size", type=float, default = 0.0)
    # parser.add_argument("-load_allres", type=bool, default = False)
    
    os.makedirs(save_path,exist_ok=True)
    
    args = parser.parse_args(sys.argv[1:])
    with open(os.path.join(save_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    
    blocks_info_json_path = os.path.join(blocks_path,"blocks_info.json")
    with open(blocks_info_json_path, 'r') as file:
        blocks_info = json.load(file)
    blocks_num = len(blocks_info)
    
    #TODO:for test
    # blocks_num = 2
    
    #2 cameras.json
    block_path = []
    all_camera_info = []
    for i in range(blocks_num):
        path_ = os.path.join(optimizer_pcds_path,f"block{i}")
        block_path.append(path_)
        block_camerainfo_json_path = os.path.join(path_,"cameras.json")
        with open(block_camerainfo_json_path, 'r') as file:
            block_camera_info = json.load(file)
            all_camera_info += block_camera_info
    #filter all_camera_info to remove repeated images
    all_camera_info = remove_duplicates(all_camera_info,"img_name")
    
    camera_json_save_path = os.path.join(save_path,"cameras.json")
    with open(camera_json_save_path, "w") as file:
        json.dump(all_camera_info, file)
        
    #3 point_cloud
    orgin_ranges = []
    for i in range(blocks_num):
        orgin_ranges.append(blocks_info[i]["orgin_block_range"])
    
    #save pcd in origin_extend
    for i in tqdm(range(blocks_num)):
        iter_num = 30000
        gaussians = GaussianModel(args.sh_degree)
        block_pcd_path = os.path.join(block_path[i],"point_cloud",f"iteration_{iter_num}","point_cloud.ply")
        gaussians.load_ply(block_pcd_path)
        
        bbox = orgin_ranges[i]
        origin_save_path = os.path.join(block_path[i],"point_cloud",f"iteration_{iter_num}","origin_point_cloud.ply")
        gaussians.save_ply_origin(origin_save_path,bbox,block_index=i,x_block_num=2,z_block_num=2)
        # gaussians.save_ply_origin(origin_save_path,bbox,block_index=i,x_block_num=4,z_block_num=4)
        # gaussians.save_ply_origin(origin_save_path,bbox,block_index=i,x_block_num=4,z_block_num=2)
        
    
    for iter_num in iter_nums:
        pcd_save_path = os.path.join(save_path,"point_cloud",f"iteration_{iter_num}")
        os.makedirs(pcd_save_path,exist_ok=True)

        vertex = []
        for i in tqdm(range(blocks_num)):
        # for i in range(blocks_num):
            #获取每个block的pcd
            block_pcd_path = os.path.join(block_path[i],"point_cloud",f"iteration_{iter_num}","origin_point_cloud.ply")
            # block_pcd_path = os.path.join(block_path[i],"point_cloud",f"iteration_{iter_num}","point_cloud.ply")

            block_ply = PlyData.read(block_pcd_path)
            vertex_data = block_ply['vertex'].data
            
            if not i :
                vertex = vertex_data
            else:
                vertex = np.concatenate((vertex, vertex_data))
        el = PlyElement.describe(vertex, 'vertex')
        ply_save_path = os.path.join(pcd_save_path,"point_cloud.ply")
        PlyData([el]).write(ply_save_path)

if __name__ == "__main__":
    data_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js"
    blocks_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js/blocks_manh"
    optimizer_pcds_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js"
    save_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js/merge"
    
    block_mege(data_path,blocks_path,optimizer_pcds_path,save_path)