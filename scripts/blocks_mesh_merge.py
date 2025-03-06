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
def mesh_merge(root_dir,train_result_dir,save_dir,iters,binary_num = 8):
    
    block_dirs = os.listdir(train_result_dir)
    
    block_num = 0
    
    for i in range(len(block_dirs)):
        if block_dirs[i].startswith("block"):
            block_num += 1
    print(block_num)
    
    # block_json_path = os.path.join(root_dir,"blocks_info.json")
    # block_info_map = {}
    # with open(block_json_path,"r") as file:
    #     block_info = json.load(file)
    #     for i in range(len(block_info)):
    #         block_info_map[block_info[i]["block_id"]] = block_info[i]
    merge_mesh = None
    for iter in iters:
        for i in tqdm(range(block_num)):
            one_block_result_path = os.path.join(train_result_dir,"block" + str(i))
            fuse_path = os.path.join(one_block_result_path,"test","ours_" + str(iter),"fusion",f"mesh_binary_search_{binary_num - 1}.ply")
            if not os.path.exists(fuse_path):
                print("mesh not found!")
                exit(0)
            # plydata = PlyData.read(fuse_path)
            block_mesh=o3d.io.read_triangle_mesh(fuse_path)
            # vertices = np.asarray(block_mesh.vertices)
            # triangles = np.asarray(block_mesh.triangles)
            
            block_mesh.merge_close_vertices(0.005)
            # extend = block_info_map[i]["orgin_block_range"]
            
            # delete_mask = ((vertices[...,0] > extend[0]) * (vertices[...,0] < extend[2]) * (vertices[...,2] > extend[1]) * (vertices[...,2] < extend[3]))
            # delete_vertices_index = np.where(~delete_mask)[0]
            
            #filter mesh in bound
            # np.isin(triangles[...,2],delete_vertices_index)
            # delete_triangle_mask = np.any(np.isin(triangles, delete_vertices_index), axis=-1)
            # delete_triangle_index = np.where(delete_triangle_mask)[0]
            
            #delete vertices and mesh
            # block_mesh.vertices = o3d.utility.Vector3dVector(np.delete(vertices, delete_vertices_index, axis=0))
            # block_mesh.triangles = o3d.utility.Vector3iVector(np.delete(triangles, delete_triangle_index, axis=0))     
            
            #优化
            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            #     triangle_clusters, cluster_n_triangles, cluster_area = (block_mesh.cluster_connected_triangles())

            # triangle_clusters = np.asarray(triangle_clusters)
            # cluster_n_triangles = np.asarray(cluster_n_triangles)
            # cluster_area = np.asarray(cluster_area)
            # n_cluster = np.sort(cluster_n_triangles.copy())[-50]
            # n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
            # triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            # block_mesh.remove_triangles_by_mask(triangles_to_remove)
            # block_mesh.remove_unreferenced_vertices()
            # block_mesh.merge_close_vertices(0.005)
            # block_mesh.remove_degenerate_triangles()

            if merge_mesh is None:
                merge_mesh = block_mesh
            else:
                #merge_mesh
                merge_vertices_num = len(np.asarray(merge_mesh.vertices))
                merge_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate((np.asarray(merge_mesh.vertices),np.asarray(block_mesh.vertices)),axis=0))
                block_triangles = np.asarray(block_mesh.triangles) + merge_vertices_num
                merge_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate((np.asarray(merge_mesh.triangles),block_triangles), axis=0)) 
                merge_mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(merge_mesh.vertex_colors),np.asarray(block_mesh.vertex_colors)),axis=0)) 
            # merge_mesh.merge_close_vertices(0.005)

        #save
        merge_mesh.remove_unreferenced_vertices()
        # merge_mesh.merge_close_vertices(0.005)
        merge_mesh.remove_degenerate_triangles()
        
        os.makedirs(save_dir,exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(save_dir,str(iter)+"_merge.ply"), merge_mesh)  
        print("Done")  

if __name__ == "__main__":
    root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js"
    train_result_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js"
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/js_merge"
    iters = [30000]
    mesh_merge(root_dir,train_result_dir,save_dir,iters)