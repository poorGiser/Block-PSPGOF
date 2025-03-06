'''
合并多个block的mesh
'''
import os
import json
# import pymeshlab as ml
from plyfile import PlyData,PlyElement
import numpy as np
import open3d as o3d
from tqdm import tqdm
def mesh_merge(root_dir,train_result_dir,save_dir,iters,binary_num = 8,lod_level = 5):
    
    block_dirs = os.listdir(train_result_dir)
    
    block_num = 0
    
    for i in range(len(block_dirs)):
        if block_dirs[i].startswith("block"):
            block_num += 1
    print(block_num)
    
    for iter in iters:
        for level in tqdm(range(lod_level)):
            merge_mesh = None
            for i in tqdm(range(block_num)):
                one_block_result_path = os.path.join(train_result_dir,"block" + str(i))
                fuse_path = os.path.join(one_block_result_path,"test","ours_" + str(iter),"fusion",f"lod_mesh_{level}.ply")
                if not os.path.exists(fuse_path):
                    print("mesh not found!")
                    exit(0)
                    
                block_mesh=o3d.io.read_triangle_mesh(fuse_path)
                block_mesh.merge_close_vertices(0.005)

                if merge_mesh is None:
                    merge_mesh = block_mesh
                else:
                    #merge_mesh
                    merge_vertices_num = len(np.asarray(merge_mesh.vertices))
                    merge_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate((np.asarray(merge_mesh.vertices),np.asarray(block_mesh.vertices)),axis=0))
                    block_triangles = np.asarray(block_mesh.triangles) + merge_vertices_num
                    merge_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate((np.asarray(merge_mesh.triangles),block_triangles), axis=0)) 
                    merge_mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(merge_mesh.vertex_colors),np.asarray(block_mesh.vertex_colors)),axis=0)) 

            #save
            merge_mesh.remove_unreferenced_vertices()
            merge_mesh.merge_close_vertices(0.005)
            merge_mesh.remove_degenerate_triangles()
            
            os.makedirs(save_dir,exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(save_dir,"lod_mesh_" + str(level)+".ply"), merge_mesh)  
    print("Done")  

if __name__ == "__main__":
    root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station"
    train_result_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod"
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/power_station_lod_2"
    iters = [30000]
    lod_level = 5
    mesh_merge(root_dir,train_result_dir,save_dir,iters,lod_level)