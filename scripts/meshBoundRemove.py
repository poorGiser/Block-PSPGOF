'''
mesh边缘去除
'''
import open3d as o3d
import os
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pymeshlab

from matplotlib.patches import Polygon
def meshBoundRemove(root_path,mesh,save_path,bound = 0.05):
    #读取原始点云
    orgin_pcd_path = os.path.join(root_path,"sparse","0","points3D.ply")
    orgin_sparse_pcd = o3d.io.read_point_cloud(orgin_pcd_path,print_progress = True)
    orgin_sparse_pcd,_ = orgin_sparse_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=3.0)
    
    orgin_sparse_pcd_np =  np.asarray(orgin_sparse_pcd.points)
    orgin_sparse_pcd_plane = orgin_sparse_pcd_np[:,[0,2]]
    
    center = np.mean(orgin_sparse_pcd_plane, axis=0)
    scale_factor = (1 - bound)
    orgin_sparse_pcd_plane_small = center + scale_factor * (orgin_sparse_pcd_plane - center)
    
    orgin_hull = ConvexHull(orgin_sparse_pcd_plane_small)
    hull_index = np.asarray(orgin_hull.vertices)
    hull_pos = orgin_sparse_pcd_plane_small[hull_index]

    # 创建一个Matplotlib的Figure和Axes
    fig, ax = plt.subplots()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    # 创建多边形对象并添加到Axes
    polygon = Polygon(hull_pos, closed=True, edgecolor='b', facecolor='none')
    ax.add_patch(polygon)

    # plt.savefig("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/residence_merge/test.jpg")
    
    mesh_pcd_np = np.asarray(mesh.vertices)
    mesh_pcd_np_plane = mesh_pcd_np[:,[0,2]]
    
    max_point = np.max(mesh_pcd_np_plane,axis=0)
    min_point = np.min(mesh_pcd_np_plane,axis=0)
    bbox_pos = np.asarray([
        [min_point[0],min_point[1]],
        [max_point[0],min_point[1]],
        [max_point[0],max_point[1]],
        [min_point[0],max_point[1]]
    ])
    
    bbox_polygon = Polygon(bbox_pos, closed=True, edgecolor='r', facecolor='none')
    ax.add_patch(bbox_polygon)
    # plt.savefig("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/campus_merge/test.jpg")
    
    tolerance=1e-12
    points_mask = np.all((np.dot(orgin_hull.equations[:, :-1], mesh_pcd_np_plane.T) + orgin_hull.equations[:, -1][:, None] <= tolerance), axis=0)
    
    remove_mask = ~points_mask
    print(mesh)
    mesh.remove_vertices_by_mask(remove_mask)
    print(mesh)
    
    ms = pymeshlab.MeshSet()
    vertices = np.asarray(mesh.vertices,dtype=np.float64)
    vertices_num = vertices.shape[0]
    m = pymeshlab.Mesh(
    vertex_matrix=vertices,
    face_matrix=np.asarray(mesh.triangles,dtype=np.int32),
    v_color_matrix=np.concatenate((np.asarray(mesh.vertex_colors,dtype=np.float64),np.ones((vertices_num,1),dtype=np.float64)),axis= -1)
    )
    ms.add_mesh(m, "mesh")
    # ms.meshing_close_holes()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 100000,removeunref  = True)
    
    ms.save_current_mesh(save_path)
    
    print("Done!")
    
    
if __name__ == "__main__":
    root_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js"
    
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/js_merge"
    # os.makedirs(save_dir, exist_ok=True)
    
    mesh_path = save_dir + "/30000_merge.ply"
    save_path = save_dir + "/30000_merge_filtered.ply"
    
    mesh = o3d.io.read_triangle_mesh(mesh_path,print_progress = True)
    
    # meshBoundRemove(root_path=root_path,mesh=mesh,save_path = save_path,bound=0.2)
    meshBoundRemove(root_path=root_path,mesh=mesh,save_path = save_path,bound=0.5)
    
    # meshBoundRemove(root_path=root_path,mesh=mesh,save_path = save_path,bound=0.1)
    
    