import open3d as o3d
import json
import numpy as np
'''
将mesh点云的坐标进行变换到原始坐标下
'''
def our_pcd_transform(pcd_path, transform_path,save_path):
    mesh = o3d.io.read_triangle_mesh(pcd_path)
    vertices = np.asarray(mesh.vertices)
    
    #转换坐标
    point_x = vertices[...,0]
    point_y = vertices[...,1]
    point_z = vertices[...,2]
    with open(transform_path) as file:
        transform = json.load(file)
    translation = transform["translations"]
    scale = transform["scales"]
    
    translation = np.asarray(translation)
    scale = np.asarray(scale)
    
    point_x = (point_x / scale[0]) - translation[0]
    point_y = (point_y / scale[1]) - translation[1]
    point_z = (point_z / scale[2]) - translation[2]
    
    #转换位置
    new_x = point_x
    new_y = point_z
    new_z = -point_y
    
    T2 = [397325.000000,2608884.000000,0.00000]
    new_x = new_x - T2[0]
    new_y = new_y - T2[1]
    new_z = new_z - T2[2]
    
    vertices[...,0] = new_x
    vertices[...,1] = new_y
    vertices[...,2] = new_z

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(save_path, mesh)

if __name__ == "__main__":
    pcd_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/power_station_lod_2/lod_mesh_4.ply"
    json_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/transform.json"
    save_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/power_station_lod_3/lod_mesh_4_transform.ply"
    
    our_pcd_transform(pcd_path, json_path,save_path)