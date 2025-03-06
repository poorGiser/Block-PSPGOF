import open3d as o3d
import json
import numpy as np
def raw_pcd_transform(pcd_path, transform_path,save_path):
    point_cloud = o3d.io.read_point_cloud(pcd_path)
    print(point_cloud)
    
    points = np.asarray(point_cloud.points)#n*3
    points_x = points[:,0]
    points_y = points[:,1]
    points_z = points[:,2]
    
    #互换yz
    temp = np.ones_like(points_y) * points_y
    points_y = -points_z
    points_z = temp
    
    with open(transform_path) as file:
        transform = json.load(file)
    translation = transform["translations"]
    scale = transform["scales"]
    
    translation = np.asarray(translation)
    scale = np.asarray(scale)
    
    points_x = (points_x + translation[0]) * scale[0]
    points_y = (points_y + translation[1]) * scale[1]
    points_z = (points_z + translation[2]) * scale[2]
    
    new_points = np.stack([points_x, points_y, points_z], axis=1)
    
    point_cloud.points = o3d.utility.Vector3dVector(new_points)
    o3d.io.write_point_cloud(save_path, point_cloud)

if __name__ == "__main__":
    pcd_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/point_cloud/raw.pcd"
    json_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/transform.json"
    save_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/point_cloud/raw_transform.ply"
    
    raw_pcd_transform(pcd_path, json_path,save_path)