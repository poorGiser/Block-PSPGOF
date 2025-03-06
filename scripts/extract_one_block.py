'''
在提取mesh时，容易oom，因此提取出场景的一个block，在一个block上验证效果
'''
import os
import json
import open3d as o3d
import numpy as np
def extract_one_block(root_path,block_index):
    save_dir = root_path + "_block"+str(block_index)
    os.makedirs(save_dir,exist_ok=True)
    
    with open(os.path.join(root_path,"blocks_manh","blocks_info.json")) as f:
        block_infos = json.load(f)
    for block_info in block_infos:
        if block_info["block_id"] == block_index:
            orgin_block_range = block_info["orgin_block_range"]
            image_names = block_info["image_names"]
    raw_ply = os.path.join(root_path,"point_cloud","raw_transform.ply")
    point_cloud = o3d.io.read_point_cloud(raw_ply)
    #point_cloud中提取出orgin_block_range内部的点云
    point_cloud_np = np.asarray(point_cloud.points)
    mask = (point_cloud_np[...,0] > orgin_block_range[0]) * (point_cloud_np[...,0] < orgin_block_range[2]) * (point_cloud_np[...,2] > orgin_block_range[1]) * (point_cloud_np[...,2] < orgin_block_range[3])
    point_cloud_np = point_cloud_np[mask]
    
    pcd_save_path = os.path.join(save_dir,"point_cloud")
    os.makedirs(pcd_save_path,exist_ok=True)
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)
    o3d.io.write_point_cloud(os.path.join(pcd_save_path,"raw_transform.ply"), new_point_cloud)
    
    #提取出所有的图片
    image_save_dir = os.path.join(save_dir,"images")
    os.makedirs(image_save_dir,exist_ok=True)
    for image_name in image_names:
        image_path = os.path.join(root_path,"images",image_name+".jpg")
        os.system(f"cp {image_path} {image_save_dir}")
    
    #提取出ply
    ply_save_path = os.path.join(save_dir,"sparse","0")
    os.makedirs(ply_save_path,exist_ok=True)
    ply_path=os.path.join(root_path,"blocks_manh",str(block_index),f"pcd_{block_index}_cov.ply")
    #复制并重命名
    os.system(f"cp {ply_path} {ply_save_path}")
    
    #将.bin文件复制并保存
    # cameras_path = os.path.join(root_path,"sparse","0","cameras.bin")
    cameras_path = os.path.join(root_path,"sparse","0","cameras.txt")
    
    camera_save_path = os.path.join(save_dir,"sparse","0")
    os.system(f"cp {cameras_path} {camera_save_path}")
    
    # images_bin = os.path.join(root_path,"sparse","0","images.bin")
    images_bin = os.path.join(root_path,"sparse","0","images.txt")
    
    images_save_path = os.path.join(save_dir,"sparse","0")
    os.system(f"cp {images_bin} {images_save_path}")
    
    # points3D_bin = os.path.join(root_path,"sparse","0","points3D.bin")
    points3D_bin = os.path.join(root_path,"sparse","0","points3D.txt")
    
    points3D_save_path = os.path.join(save_dir,"sparse","0")
    os.system(f"cp {points3D_bin} {points3D_save_path}")
    
    val_images_json_path = os.path.join(root_path,"val_images.json")
    val_images_save_path = os.path.join(save_dir,"val_images.json")
    os.system(f"cp {val_images_json_path} {val_images_save_path}")
    
    print("Done!")
    
if __name__ == "__main__":
    root_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/building"
    # save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data"
    
    block_index = 1
    extract_one_block(root_path,block_index)



