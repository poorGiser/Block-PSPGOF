import torch
import os
import open3d as o3d

# 创建一个简单的三角形mesh
# vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
# triangles = [[0, 1, 2]]
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)
# render_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble2/block0/test/ours_30000/fusion"
# cells = torch.load(os.path.join(render_path, "cells.pt"))

# n = 100
# a = torch.ones((1,4096),dtype=torch.bool).reshape(-1)
# occ_n = a > 0
# b = torch.ones((n,4),dtype=torch.long)

# occ_fx4 = a[b.reshape(-1)].reshape(-1, 4)

# print(1)

# sphinx_gallery_thumbnail_number = 2
# import pymeshlab
# ms = pymeshlab.MeshSet()
# ms.load_new_mesh("gaussian-opacity-fields-main/output/rubble6/block0/test/ours_30000/fusion/mesh_binary_search10_7.ply")
# print(ms.mesh_number())
# print(ms.current_mesh().vertex_number())


# ms.meshing_close_holes()
# ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 300)

# ms.save_current_mesh("gaussian-opacity-fields-main/output/rubble6/block0/test/ours_30000/fusion/mesh_binary_search10_7_processed.ply")
# import trimesh

# # 创建一个示例网格
# mesh = trimesh.creation.box()

# # 假设我们为网格设置了颜色
# # mesh.visual.vertex_colors = trimesh.color.random_color(mesh.vertices)

# # 获取颜色信息
# colors = mesh.visual.vertex_colors

# # 打印颜色信息
# print(colors)
# ms.save_current_mesh(output_path + "convex_hull.obj")


# import cv2
# import numpy as np
 
# def illum(img):
#     iso1 = 240
#     img_bw1 = np.any(img > iso1,axis=-1)
    
#     img_bw2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imwrite("GRAY.png",img_bw2)
    
#     thresh = cv2.threshold(img_bw2 , 238, 255, 0)[1]
    
#     thresh = (np.logical_or(img_bw1,thresh) * 255).astype(np.uint8)
#     cv2.imwrite("area.png",thresh)
    
#     mask = (thresh != 0)
    
#     h,w = img.shape[0],img.shape[1]
#     mask = mask[...].reshape(h*w)
#     img = img.reshape(h*w,3)
#     index = np.where(mask)[0]
#     img[index] = img[index] // np.array([[1.35,1.25,1.2]])
    
    
#     img = img.reshape(h,w,3)
#     return img

# test_img = cv2.imread("/home/chenyi/gaussian-splatting/data/residence/images/DJI_0218.JPG")

# process_img = illum(test_img)
# cv2.imwrite("test.png",process_img)

# print("done!")
# from scene import Scene, GaussianModel

# output_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble11"

# iter = 30000


# pcd_path = os.path.join(output_dir,"block0","point_cloud",f"iteration_{iter}","point_cloud.ply")
# gaussians = GaussianModel(3)

# # plane_iso = 1.5
# # plane_iso = 5.0
# # plane_iso = 5.0
# plane_iso = 2.0



# gaussians.load_ply(pcd_path)
# scaling = gaussians.get_scaling
# xyz = gaussians.get_xyz

# sorted_scaling = torch.sort(scaling,dim=1)[0]


# bs = (sorted_scaling[...,1] / sorted_scaling[...,0])
# plane_mask = (bs > plane_iso)
# plane_points = xyz[plane_mask]
# ball_points = xyz[~plane_mask]

# temp_save_path = "/home/chenyi/gaussian-opacity-fields-main/temp"
# plane_point_cloud = o3d.geometry.PointCloud()
# plane_point_cloud.points = o3d.utility.Vector3dVector(plane_points.detach().cpu().numpy())
# o3d.io.write_point_cloud(os.path.join(temp_save_path,f"plane_{plane_iso}_2_pro.ply"), plane_point_cloud, write_ascii=False)
# ball_point_cloud = o3d.geometry.PointCloud()
# ball_point_cloud.points = o3d.utility.Vector3dVector(ball_points.detach().cpu().numpy())
# o3d.io.write_point_cloud(os.path.join(temp_save_path,f"ball_{plane_iso}_2_pro.ply"), ball_point_cloud, write_ascii=False)

# print("dONE!")

# import torch

# # 假设我们有一个 n * 3 的张量
# n = 5  # 例如，5个点
# tensor_n3 = torch.tensor([[1, 2, 3], 
#                            [4, 5, 6], 
#                            [7, 8, 9], 
#                            [10, 11, 12],
#                            [13, 14, 15]], dtype=torch.float)

# # 假设我们有一个 n * 1 的索引张量，取值为 0 到 2
# index_tensor = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long).reshape(-1,1)

# # 使用 index_select 从 tensor_n3 中的每行选取 index_tensor 中的索引位置的元素
# # dim=0 表示沿着行的方向选择，即选择特定行
# selected_values = tensor_n3.index_select(dim=0, index=index_tensor)

# # selected_values 现在是一个 n * 1 的张量
# print(selected_values)

# a = torch.randn((5,4))

# index = torch.tensor([[0,1],[2,3]])

# print(1)

# import numpy as np

# # 假设n是矩阵a和b的行数，这里我们用一个具体的例子
# n = 5  # 例如5行

# # 创建一个n*3的矩阵a，随机填充一些值
# a = np.random.randint(0, 10, size=(n, 3))

# # 创建一个n*1的矩阵b，填充0-2之间的随机整数
# b = np.random.randint(0, 3, size=(n, 1))

# # 将a中对应b的索引位置的元素置为0
# a[np.arange(n), b] = 0

# print("修改前的矩阵a:\n", a)
# print("索引矩阵b:\n", b)
# print("修改后的矩阵a:\n", a)


# import numpy as np
# from scipy.ndimage import gaussian_filter

# def gaussian_filter_point_cloud(point_cloud, sigma=1.0):
#     """
#     Apply Gaussian filter to a point cloud.

#     Parameters:
#     - point_cloud: numpy array of shape (N, 3) containing 3D points
#     - sigma: standard deviation for Gaussian kernel

#     Returns:
#     - smoothed_point_cloud: numpy array of same shape as input, with smoothed points
#     """
#     smoothed_point_cloud = np.zeros_like(point_cloud)
#     for dim in range(3):  # Apply filter along each dimension separately
#         smoothed_point_cloud[:, dim] = gaussian_filter(point_cloud[:, dim], sigma=sigma, mode='constant')
#     return smoothed_point_cloud

# # Example usage:
# if __name__ == "__main__":
#     # Generate a random point cloud for demonstration
#     np.random.seed(0)
#     point_cloud = np.random.rand(100, 3) * 100  # 100 points in 3D space
    
#     # Apply Gaussian filter
#     sigma = 2.0
#     smoothed_point_cloud = gaussian_filter_point_cloud(point_cloud, sigma=sigma)
    
#     # Print example results
#     print("Original Point Cloud:")
#     print(point_cloud[:5])  # Print first 5 points
#     print("\nSmoothed Point Cloud (sigma={}):".format(sigma))
#     print(smoothed_point_cloud[:5])  # Print first 5 smoothed points


# import numpy as np
# import open3d as o3d

# # 加载PLY文件
# mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble/block0/test/ours_30000/fusion/mesh_binary_search_7.ply")
# mesh = mesh.remove_degenerate_triangles()
# mesh = mesh.remove_non_manifold_edges()
# # 计算每个面的面积
# def compute_area(v0, v1, v2):
#     cross_product = torch.cross(v1 - v0, v2 - v0,dim=1)
#     return 0.5 * torch.norm(cross_product,dim=1)

# # 设置面积阈值
# # area_threshold = 0.25  # 这个值需要根据你的具体需求来设置
# # area_threshold = 0.125  
# area_threshold = 1.0



# # 删除面积较大的面
# large_faces = []

# triangles = torch.tensor(np.asarray(mesh.triangles)).to(torch.int64)
# vertices = torch.tensor(np.asarray(mesh.vertices))

# mean_pos = torch.mean(vertices,dim=0) * 1

# tri_points = vertices[triangles]
# tri_areas = compute_area(tri_points[...,0],tri_points[...,1],tri_points[...,2])

# # tri_mask = (tri_areas > area_threshold) * (torch.any(tri_points[...,:,1] < mean_pos[1],dim = 1))
# # tri_mask = ((tri_areas > area_threshold) * (torch.all(tri_points[...,:,1] < mean_pos[1] * 0.8,dim = 1)))
# tri_mask = (tri_areas > area_threshold)


# mesh.remove_triangles_by_mask(tri_mask.numpy())
# mesh.remove_unreferenced_vertices()

# # 保存修改后的网格
# o3d.io.write_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble/block0/temp/test.ply", mesh)
# print(1)


# from utils.image_utils import psnr
# import numpy as np
# import cv2

# def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
#   if img.shape[-1] != ref.shape[-1]:
#     raise ValueError(
#         f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
#     )
#   num_channels = img.shape[-1]
#   img_mat = img.reshape([-1, num_channels])
#   ref_mat = ref.reshape([-1, num_channels])
#   is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
#   mask0 = is_unclipped(img_mat)
#   # Because the set of saturated pixels may change after solving for a
#   # transformation, we repeatedly solve a system `num_iters` times and update
#   # our estimate of which pixels are saturated.
#   for _ in range(num_iters):
#     # Construct the left hand side of a linear system that contains a quadratic
#     # expansion of each pixel of `img`.
#     a_mat = []
#     for c in range(num_channels):
#       a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
#     a_mat.append(img_mat)  # Linear term.
#     a_mat.append(np.ones_like(img_mat[:, :1]))  # Bias term.
#     a_mat = np.concatenate(a_mat, axis=-1)
#     warp = []
#     for c in range(num_channels):
#       # Construct the right hand side of a linear system containing each color
#       # of `ref`.
#       b = ref_mat[:, c]
#       # Ignore rows of the linear system that were saturated in the input or are
#       # saturated in the current corrected color estimate.
#       mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
#       ma_mat = np.where(mask[:, None], a_mat, 0)
#       mb = np.where(mask, b, 0)
#       # Solve the linear system. We're using the np.lstsq instead of np because
#       # it's significantly more stable in this case, for some reason.
#       w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
#       assert np.all(np.isfinite(w))
#       warp.append(w)
#     warp = np.stack(warp, axis=-1)
#     # Apply the warp to update img_mat.
#     img_mat = np.clip(
#         np.matmul(a_mat, warp), 0, 1)
#   corrected_img = np.reshape(img_mat, img.shape)
#   return corrected_img


# # color_correct(rendering['rgb'], gt_rgb)
# if __name__ == "__main__":
#   gt_image = cv2.imread('/home/chenyi/multinerf-main/testImage/gt.png') / 255
#   pred_image = cv2.imread('/home/chenyi/multinerf-main/testImage/pred.png') / 255
  
#   psnr1 = psnr(torch.tensor(pred_image.transpose(2,0,1)), torch.tensor(gt_image.transpose(2,0,1))).mean().double()
#   print(psnr1)
  
#   pred_image_cc = color_correct(pred_image,gt_image)
#   #保存pred_image_cc
#   cv2.imwrite('/home/chenyi/multinerf-main/testImage/pred_cc.png',pred_image_cc * 255)
  
#   psnr2 = psnr(torch.tensor(pred_image_cc.transpose(2,0,1)), torch.tensor(gt_image.transpose(2,0,1))).mean().double()
  
#   print(psnr2)
  
  
# torch.cuda.set_device(torch.device(f"cuda:1"))
# a = torch.tensor(1,device="cuda")
# print(a)


# import open3d as o3d

# # 读取点云数据
# point_cloud = o3d.io.read_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/building/p_f/points3D.ply")


# # 对点云进行滤波操作
# point_cloud_filtered = point_cloud.remove_statistical_outlier(nb_neighbors = 50,std_ratio=3.0,print_progress = True)[0]
# bbox = point_cloud_filtered.get_axis_aligned_bounding_box()

# o3d.io.write_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/building/p_f/test.ply", point_cloud_filtered)

# import open3d as o3d
# import pymeshlab
# ms = pymeshlab.MeshSet()
# ms.load_new_mesh('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/rubble_merge/30000_merge_filtered.ply')
# 读取点云文件
# mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/rubble_merge/30000_merge_filtered.ply",print_progress = True)

# ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 50000,removeunref  = True)
# ms.save_current_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/rubble_merge/" + "30000_merge_filtered2.ply")
# print(1)
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     triangle_clusters, cluster_n_triangles, cluster_area = (
#         mesh.cluster_connected_triangles())


# largest_cluster_idx = cluster_n_triangles.argmax()
# print(largest_cluster_idx)
# triangles_to_remove = triangle_clusters != largest_cluster_idx
# mesh.remove_triangles_by_mask(triangles_to_remove)

# o3d.io.write_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/rubble_merge/30000_merge_filtered2.ply", mesh)
# import os
# import json
# root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/residence"

# #判断每个block中的图片数目
# block_num = 8


# blocks_dir = os.path.join(root_dir,"blocks_manh")
# json_path = os.path.join(blocks_dir,"blocks_info.json")
# with open(json_path) as file:
#     block_datas = json.load(file)

# for block_data in block_datas:
#     print(block_data["block_id"],len(block_data["image_names"]))

# import open3d as o3d
# import numpy as np
# import pandas as pd

# # 读取点云文件，例如 .ply 或 .pcd 格式
# point_cloud = o3d.io.read_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/matrix_city_all/sparse/0/points3D.ply")
# print(point_cloud)

# cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
# print(cl)

# # 可视化点云
# points =  np.asarray(cl.points)

# print(1)

# o3d.io.write_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/matrix_city_all/sparse/0/pcd_remove.ply", cl)
# import os
# import json
# root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station"
# images_dir = os.path.join(root_dir,"images")
# json_path = os.path.join(root_dir,"val_images.json")
# with open(json_path) as file:
#     image_names = json.load(file)
# for image_name in image_names:
#     #复制图片到另一个文件夹
#     os.system("cp " + os.path.join(images_dir,image_name + ".jpg") + " /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/val_images/" +image_name + ".jpg")

#对数据集中的图片下采样并保存
# import os
# from PIL import Image
# import torch
# from utils.general_utils import PILtoTorch
# import cv2
# import numpy as np
# from tqdm import tqdm
# scenes = ["power_station","rural","rubble","building","residence","campus"]
# for scene in scenes:
#     root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/" + scene
#     images_dir = os.path.join(root_dir,"images")
#     image_names = os.listdir(images_dir)
    
#     save_dir = os.path.join(root_dir,"images_ds")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     for image_name in tqdm(image_names):
#         image_path = os.path.join(images_dir,image_name)
#         image = Image.open(image_path)
        
#         orig_w, orig_h = image.size
#         resolution = round(orig_w/(1 * 4)), round(orig_h/(1 * 4))
#         resized_image_rgb = PILtoTorch(image, resolution)
        
#         image_np = resized_image_rgb.permute(1,2,0).numpy()* 256
#         image_np = np.clip(image_np, 0, 256, out=image_np)
#         #rgb转bgr
#         image_np = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR) 

#         #保存图片
#         cv2.imwrite(os.path.join(save_dir,image_name),image_np)
        
#         image.close()

#筛选在bbox内部的mesh
# import open3d as o3d
# import numpy as np

# /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/point_cloud/raw_transform.ply
# raw_mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/point_cloud/raw_transform.ply")
# raw_mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/point_cloud/raw_transform.ply")

# raw_points = np.asarray(raw_mesh.vertices)#n * 3
# scene_min = np.min(raw_points, axis=0)
# scene_max = np.max(raw_points, axis=0)

# bbox = [scene_min[0],scene_min[2],scene_max[0],scene_max[2]]
#rubble
# bbox = [
#                    -0.057854454040493476,
#                     -8.563912391662598,
#                     6.75537109375,
#                     -2.5552224845885947
# ]

# mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/campus_merge/30000_merge2.ply")
# # mesh = o3d.io.read_point_cloud("/home/chenyi/SuGaR/output/refined_ply/power_station/low.ply")

# print(mesh)
# mesh.merge_close_vertices(0.005)

# mesh.remove_unreferenced_vertices()
# merge_mesh.merge_close_vertices(0.005)
# mesh.remove_degenerate_triangles()
# print(mesh)

# points = np.asarray(mesh.vertices)
# mask = ((points[...,0] < bbox[0]) | (points[...,0] > bbox[2]) | (points[...,2] < bbox[1]) | (points[...,2] > bbox[3]))
# mesh.remove_vertices_by_mask(mask)
# print(mesh)

# o3d.io.write_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/campus_block1/campus_block1-2.ply", mesh)
# print("Done!")
# pcd_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/merge3/point_cloud/iteration_30000/point_cloud.ply"
# # mesh_path = "/home/chenyi/SuGaR/output/refined_mesh/rural_block0/sugarfine_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj"

# # print("model_size",os.path.getsize(mesh_path)/1024/1024)
# pcd = o3d.io.read_point_cloud(pcd_path)
# num = np.asarray(pcd.points).shape[0]
# print("gaussuan_num",num)

# import cv2
# import numpy as np
# import cv2
# import numpy as np

# import cv2
# import numpy as np

# # 1. 加载图像
# image = cv2.imread('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble/high_light/gray/000050_processed.png')
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 2. 创建掩膜（这里用简单的阈值处理作为示例）
# mask = cv2.imread('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble/high_light/gray/000050_mask.png')

# mask = mask[:,:,0]

# mask = np.clip(mask, 0, 1)

# # 3. 查找轮廓
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 4. 在原图上绘制轮廓
# # -1 表示填充所有轮廓，颜色为绿色（BGR格式），厚度为2
# cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

# # 显示结果
# cv2.imwrite('/home/chenyi/gaussian-opacity-fields-main/overlay_result.jpg', image)



# 释放资源
# cv2.destroyAllWindows()
# # 读取图像
# image = cv2.imread('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble/high_light/gray/000050.png')

# # 创建一个掩膜（这里假设掩膜是二值图像）
# mask = cv2.imread('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble/high_light/gray/000050_mask.png', cv2.IMREAD_GRAYSCALE)

# # 找到掩膜的边界
# edges = cv2.Canny(mask, 100, 200)

# # 创建一个颜色图像以便于绘制边界
# color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# # 在原图像上绘制边界
# result = cv2.addWeighted(image, 1, color_edges, 1, 0)

# # 保存结果
# cv2.imwrite('result.jpg', result)

# # 释放资源
# cv2.destroyAllWindows()
# import numpy as np
# def non_uniform_sampling(start, end, levels):
#     # 使用指数曲线生成非均匀采样
#     x = np.linspace(0, 1, levels)  # 从0到1线性分布
#     samples = start + (end - start) * (x**2)  # 使用指数2进行加速
    
#     return samples

# print(1)
# import open3d as o3d
# import numpy as np

# # 创建一个简单的三角形网格
# mesh = o3d.geometry.TriangleMesh()

# # 定义顶点位置
# vertices = np.array([[0, 0, 0], 
#                      [1, 0, 0], 
#                      [0, 1, 0], 
#                      [0, 0, 1]])

# # 定义三角形面
# triangles = np.array([[0, 1, 2], 
#                       [0, 1, 3], 
#                       [0, 2, 3], 
#                       [1, 2, 3]])

# # 设置网格的顶点和面
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)

# # 设置每个顶点的颜色（RGB格式，值在0到1之间）
# vertex_colors = np.array([[1, 0, 0],  # 顶点0: 红色
#                           [0, 1, 0],  # 顶点1: 绿色
#                           [0, 0, 1],  # 顶点2: 蓝色
#                           [1, 1, 0]]) # 顶点3: 黄色

# # 设置顶点颜色
# mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# # 导出为 GLTF 格式
# o3d.io.write_triangle_mesh("output_mesh_with_colors.gltf", mesh)
# pcd_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/point_cloud/raw.pcd"
# point_cloud = o3d.io.read_point_cloud(pcd_path)
# o3d.io.write_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/point_cloud/raw.ply", point_cloud)
# import open3d as o3d

# # 读取三角网格模型
# mesh = o3d.io.read_triangle_mesh("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/mesh/power_station_lod_2/lod_mesh_0_transform.ply")

# # 获取顶点信息
# vertices = mesh.vertices

# import numpy as np

# # 将顶点转换为 NumPy 数组
# vertices_np = np.asarray(vertices)
# print(1)
from moviepy.video.io.VideoFileClip import VideoFileClip

# 加载视频文件
video = VideoFileClip("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js/merge/train/ours_30000/fly.mp4")

# 获取视频的总时长
total_duration = video.duration

# 剪掉前5秒
video_clip = video.subclip(12, total_duration)

# 保存剪辑后的视频
video_clip.write_videofile("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js/merge/train/ours_30000/fly_short.mp4", codec="libx264")



        
        
        
        