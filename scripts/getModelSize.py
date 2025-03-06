#输出训练结果中的平均模型大小
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":
    root_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rural"
    block_names = os.listdir(root_path)
    block_names = [name for name in block_names if "block" in name]
    
    block_nums = len(block_names)
    
    med_sizes = []
    model_sizes = []
    g_nums = []
    for block_name in tqdm(block_names):
        block_path = os.path.join(root_path, block_name)

        one_med_path = os.path.join(block_path, "test","ours_30000","fusion","cells.pt")
        model_path = os.path.join(block_path, "test","ours_30000","fusion","mesh_binary_search_7.ply")
        gaussian_path = os.path.join(block_path,"point_cloud","iteration_30000","point_cloud.ply")
        
        pcd = o3d.io.read_point_cloud(gaussian_path)
        g_nums.append(np.asarray(pcd.points).shape[0])
        one_med_size = os.path.getsize(one_med_path)
        model_size = os.path.getsize(model_path)
        
        #转换成MB
        one_med_size = one_med_size / 1024 / 1024
        model_size = model_size / 1024 / 1024
        
        med_sizes.append(one_med_size)
        model_sizes.append(model_size)
        
    #计算平均大小
    print("平均med大小为：", sum(med_sizes) / block_nums)
    print("平均模型大小为：", sum(model_sizes) / block_nums)
    print("平均Gaussian数目为：", sum(g_nums) / block_nums)
    