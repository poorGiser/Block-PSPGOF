#block for large drone scene
import os
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras,storePly,fetchPly
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scene.gaussian_model import BasicPointCloud
from scene import Scene, GaussianModel
from utils.graphics_utils import getProjectionMatrix
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm import tqdm
import json
def remove_outliers(points, threshold=3):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    z_scores = np.abs((points - mean) / std)
    outliers = np.where(z_scores > threshold)
    filtered_points = np.delete(points, outliers[0], axis=0)
    return filtered_points

def remove_outliers_pcn(pcd, threshold=3):
    points = pcd.points
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    z_scores = np.abs((points - mean) / std)
    outliers = np.where(z_scores > threshold)
    
    filtered_points = np.delete(pcd.points, outliers[0], axis=0)
    filtered_colors =  np.delete(pcd.colors, outliers[0], axis=0)
    filtered_normals =  np.delete(pcd.normals, outliers[0], axis=0)
    return BasicPointCloud(filtered_points,filtered_colors,filtered_normals)
def ndc2Pix(v,S):
	return ((v + 1.0) * S - 1.0) * 0.5
def cross_product(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def is_convex_polygon(polygon):
    n = len(polygon)
    direction = None
    
    for i in range(n):
        p1, p2, p3 = polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]
        cp = cross_product(p1, p2, p3)
        
        if cp != 0:
            if direction is None:
                direction = cp > 0
            elif direction != (cp > 0):
                return False
    
    return True
def scence_block(colmap_data_path,x_block_num,z_block_num,val_names,bound_extend,visibility_threshold,near,far,invalid_image_names = None):
    #get sparse info
    sparse_path = os.path.join(colmap_data_path,"sparse","0")
    if not os.path.exists(sparse_path):
        print("sparse result not found!")
        return
    #创建分割文件夹
    split_block_path = os.path.join(colmap_data_path,"blocks_manh")
    # split_block_path = os.path.join(colmap_data_path,"blocks_manh_1")
    
    for i in range(x_block_num * z_block_num):
        block_split_path = os.path.join(split_block_path,str(i))
        os.makedirs(block_split_path,exist_ok=True)
    bin_format = True
    cameras_info_path = os.path.join(sparse_path,"cameras.bin")
    if not os.path.exists(cameras_info_path):
        bin_format = False
    
    if bin_format:
        image_info_path = os.path.join(sparse_path,"images.bin")
        points3D_path = os.path.join(sparse_path,"points3D.bin")
    else:
        cameras_info_path = os.path.join(sparse_path,"cameras.txt")
        image_info_path = os.path.join(sparse_path,"images.txt")
        points3D_path = os.path.join(sparse_path,"points3D.txt")
    
    #read info
    if bin_format:
        #外参
        cam_extrinsics = read_extrinsics_binary(image_info_path)
        #内参
        cam_intrinsics = read_intrinsics_binary(cameras_info_path)
    else:
        cam_extrinsics = read_extrinsics_text(image_info_path)
        cam_intrinsics = read_intrinsics_text(cameras_info_path)
    
    #get all cameras info
    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(colmap_data_path, reading_dir))
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    cam_centers = []#世界坐标系下的cam坐标
    cam_w2cs = []
    for cam in cam_infos:
        if cam.image_name in invalid_image_names:
            continue
        W2C = getWorld2View2(cam.R, cam.T)
        cam_w2cs.append(W2C)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    cam_centers_np = np.asarray(cam_centers)
    cam_centers_plane = np.concatenate((cam_centers_np[:,0],cam_centers_np[:,2]),axis=-1)#
    
    #获取range
    #Camera-position-based
    camera_range = (np.min(cam_centers_plane,axis=0),np.max(cam_centers_plane,axis=0))
    camera_block_width = ((camera_range[1][0] -camera_range[0][0]) / x_block_num,(camera_range[1][1] -camera_range[0][1])  / z_block_num)
    
    ply_path = os.path.join(colmap_data_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(colmap_data_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(colmap_data_path, "sparse/0/points3D.txt")
    # ply_path = os.path.join(colmap_data_path, "sparse_off/points3D.ply")
    # bin_path = os.path.join(colmap_data_path, "sparse_off/points3D.bin")
    # txt_path = os.path.join(colmap_data_path, "sparse_off/points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    pcd_removed = remove_outliers_pcn(pcd)#remove invalid points
    # pcd_removed = remove_outliers_pcn(pcd_removed)#remove invalid points
    
    # storePly("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/matrix_city_all/sparse/0/pcd_remove.ply", pcd_removed.points, pcd_removed.colors)
    # storePly("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/residence/test/pcd.ply", pcd.points, pcd.colors)
    
    #计算点云边界
    scene_range = (np.min(pcd_removed.points,axis=0), np.max(pcd_removed.points,axis=0))
    
    #绘制点云边界和相机边界
    fig, ax = plt.subplots()
    rect1 = patches.Rectangle((camera_range[0][0], camera_range[0][1]), camera_range[1][0] - camera_range[0][0],camera_range[1][1] - camera_range[0][1] , linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((scene_range[0][0], scene_range[0][2]), scene_range[1][0] - scene_range[0][0],scene_range[1][2] - scene_range[0][2] , linewidth=1, edgecolor='b', facecolor='none')
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.scatter(cam_centers_plane[:,0], cam_centers_plane[:,1], color='blue')  # 散点的颜色设置为蓝色
    ax.grid(True)
    fig.savefig(colmap_data_path + '/rectangle_plot.png', format='png')
    
    #计算每个block的范围,[(xmin,zmin,xmax,zmax)]
    extend_block_ranges = [[] for i in range(x_block_num * z_block_num)]
    origin_block_ranges = [[] for i in range(x_block_num * z_block_num)]
    
    
    #渐进式分割场景
    #从x轴分割
    camera_num = cam_centers_plane.shape[0]
    x_split_len = []
    x_split_len.append(camera_range[0][0])
    x_add = (camera_range[1][0] - camera_range[0][0]) / 1000
    x_node_num = 0
    x_camera_num_iso = camera_num / x_block_num
    
    current_index = 0
    temp_end_x = camera_range[0][0]
    while x_node_num != (x_block_num - 1):
        current_start_x = x_split_len[current_index]
        temp_end_x += x_add
        
        current_num = np.sum((cam_centers_plane[...,0] > current_start_x) * (cam_centers_plane[...,0] < temp_end_x))
        if current_num > x_camera_num_iso:
            x_split_len.append(temp_end_x)
            x_node_num+=1
            current_index += 1
    x_split_len.append(camera_range[1][0])
    
    #从z轴分割
    z_add = (camera_range[1][1] - camera_range[0][1]) / 1000
    for i in range(x_block_num):
        start_x = x_split_len[i]
        end_x = x_split_len[i + 1]
        
        x_block_cameras = cam_centers_plane[(cam_centers_plane[...,0] > start_x) * (cam_centers_plane[...,0] < end_x)]
        z_camera_num_iso = camera_num / (x_block_num * z_block_num)
        
        z_split_len = []
        z_split_len.append(camera_range[0][1])
        z_node_num = 0
        current_index = 0
        temp_end_z = camera_range[0][1]
        
        while z_node_num != (z_block_num - 1):
            current_start_z = z_split_len[current_index]
            temp_end_z += z_add
            current_num = np.sum((x_block_cameras[...,1] > current_start_z) * (x_block_cameras[...,1] < temp_end_z))
            
            if current_num > z_camera_num_iso:
                # print(current_num)
                z_split_len.append(temp_end_z)
                z_node_num+=1
                current_index += 1
        z_split_len.append(camera_range[1][1])
        
        #extend
        x_index_min = i
        x_index_max = x_index_min + 1
        for j in range(z_block_num):
            z_index_min = j
            z_index_max = z_index_min + 1

            x_min = start_x
            x_max = end_x
            z_min = z_split_len[j]
            z_max = z_split_len[j + 1]
    
            if x_index_min == 0:
                x_min = scene_range[0][0].item()
            if x_index_max == x_block_num:
                x_max = scene_range[1][0].item()
            if z_index_min == 0:
                z_min = scene_range[0][2].item()
            if z_index_max == z_block_num:
                z_max = scene_range[1][2].item()
            
            block_index = z_index_min * x_block_num + x_index_min
            origin_block_ranges[block_index] = [x_min,z_min,x_max,z_max]
            if bound_extend > 0:
                x_min = x_min - bound_extend * camera_block_width[0]
                z_min = z_min - bound_extend * camera_block_width[1]
                x_max = x_max + bound_extend * camera_block_width[0]
                z_max = z_max + bound_extend * camera_block_width[1]
            
            extend_block_ranges[block_index] = [x_min,z_min,x_max,z_max]
    #Position-based
    # for i in range(x_block_num * z_block_num):
    #     z_index_min = i // x_block_num
    #     z_index_max = z_index_min + 1
    #     x_index_min = i % x_block_num
    #     x_index_max = x_index_min + 1
        
    #     x_min = x_index_min * camera_block_width[0] + camera_range[0][0]
    #     z_min = z_index_min * camera_block_width[1] + camera_range[0][1]
    #     x_max = x_min + camera_block_width[0]
    #     z_max = z_min + camera_block_width[1]
        
    #     if x_index_min == 0:
    #         x_min = scene_range[0][0].item()
    #     if x_index_max == x_block_num:
    #         x_max = scene_range[1][0].item()
    #     if z_index_min == 0:
    #         z_min = scene_range[0][2].item()
    #     if z_index_max == z_block_num:
    #         z_max = scene_range[1][2].item()
            
    #     origin_block_ranges.append([x_min,z_min,x_max,z_max])
    #     #边界扩张
    #     if bound_extend > 0:
    #         x_min = x_min - bound_extend * camera_block_width[0]
    #         z_min = z_min - bound_extend * camera_block_width[1]
    #         x_max = x_max + bound_extend * camera_block_width[0]
    #         z_max = z_max + bound_extend * camera_block_width[1]
        
    #     extend_block_ranges.append([x_min,z_min,x_max,z_max])
    
    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(cam_centers_plane[:,0], cam_centers_plane[:,1])
    plt.savefig(colmap_data_path + "/cameras.png")
    plt.clf()
    
    # ply_path = os.path.join(colmap_data_path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(colmap_data_path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(colmap_data_path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    
    #获取点云坐标
    pcd_points = pcd.points
    pcd_num = len(pcd_points)
    print("all points num:",pcd_num)
    pcd_colors = pcd.colors
    pcd_normals = pcd.normals
    
    # pcd_points_np = np.asarray(pcd_points)
    #将点云分配给blocks
    # point_index = ((pcd_points_np[:,2] - camera_range[0][1]) // camera_block_width[1]) * x_block_num + (pcd_points_np[:,0] - camera_range[0][0]) // camera_block_width[0]
    # #限制点云的index范围
    # point_index[point_index >= (x_block_num * z_block_num)] = -1
    # point_index[point_index < 0 ] = -1
    # print("在block外部的点数量",np.sum(point_index == -1) / len(point_index))
    # for i in range(x_block_num * z_block_num):
    #     print(i,":",np.sum(point_index == i)/ len(point_index))
    
    #点云分配
    blocks_infos = [
        {
            "block_id":i,
            "pcd":None,
            "pcd_indexs":[],
            "pcd_points":[],
            "pcd_colors":[],
            "pcd_normals":[],
            "orgin_block_range":origin_block_ranges[i],
            "extend_block_range":extend_block_ranges[i],
            "image_names":[],
            "add_images":[],
            "cam_indexs":[]
        }
        for i in range(z_block_num * x_block_num)
    ]
    
    block_num = z_block_num * x_block_num
    for i in range(block_num):
        block_x_min = blocks_infos[i]['extend_block_range'][0]
        block_z_min = blocks_infos[i]['extend_block_range'][1]
        block_x_max = blocks_infos[i]['extend_block_range'][2]
        block_z_max = blocks_infos[i]['extend_block_range'][3]
        
        # block_x_min = blocks_infos[i]['orgin_block_range'][0]
        # block_z_min = blocks_infos[i]['orgin_block_range'][1]
        # block_x_max = blocks_infos[i]['orgin_block_range'][2]
        # block_z_max = blocks_infos[i]['orgin_block_range'][3]
        
        for j in range(len(cam_centers)):
            #判断是否在block内部
            camera_pos = [cam_centers[j][0],cam_centers[j][2]]
            if camera_pos[0] >= block_x_min and camera_pos[1] >= block_z_min and camera_pos[0] <= block_x_max and camera_pos[1] <= block_z_max:
                blocks_infos[i]["image_names"].append(cam_infos[j].image_name)
                blocks_infos[i]["cam_indexs"].append(j)
    for i in range(block_num):
        print("block",i,":"+str(len(blocks_infos[i]["image_names"])) + " views")
    
    #select points
    #根据点云位置选择
    for i in range(block_num):
        block_x_min = blocks_infos[i]['extend_block_range'][0]
        block_z_min = blocks_infos[i]['extend_block_range'][1]
        block_x_max = blocks_infos[i]['extend_block_range'][2]
        block_z_max = blocks_infos[i]['extend_block_range'][3]
        
        #根据origin_block_range选择
        # block_x_min = blocks_infos[i]['orgin_block_range'][0]
        # block_z_min = blocks_infos[i]['orgin_block_range'][1]
        # block_x_max = blocks_infos[i]['orgin_block_range'][2]
        # block_z_max = blocks_infos[i]['orgin_block_range'][3]
        
        block_pcd_points = []
        block_pcd_colors = []
        block_pcd_normals = []
        
        for j in range(pcd_num):
            #判断是否在block内部
            pcd_pos = [pcd_points[j][0],pcd_points[j][2]]
            if pcd_pos[0] >= block_x_min and pcd_pos[1] >= block_z_min and pcd_pos[0] <= block_x_max and pcd_pos[1] <= block_z_max:
                block_pcd_points.append(pcd_points[j])
                block_pcd_colors.append(pcd_colors[j] * 255)
                block_pcd_normals.append(pcd_normals[j])
                
                blocks_infos[i]["pcd_indexs"].append(j)
           
        print(f"block{i} has {len(block_pcd_points)} points,contain {len(block_pcd_points)/len(pcd_points)}")
        
        #转换成BasicPointCloud
        block_pcd_points = np.stack(block_pcd_points,axis=0)
        block_pcd_colors = np.stack(block_pcd_colors,axis=0)
        block_pcd_normals = np.stack(block_pcd_normals,axis=0)
        block_pcd = BasicPointCloud(points=block_pcd_points, colors=block_pcd_colors, normals=block_pcd_normals)
    
        blocks_infos[i]["pcd"] = block_pcd
        blocks_infos[i]["pcd_points"] = block_pcd_points
        blocks_infos[i]["pcd_colors"] = block_pcd_colors
        blocks_infos[i]["pcd_normals"] = block_pcd_normals
        storePly(os.path.join(split_block_path,str(i),"pcd_" + str(i) + ".ply"), block_pcd_points, block_pcd_colors)
        
    #Visibility-based:add view
    offsets = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]
    for i in range(block_num):
        #get pcd bbox of block
        #去除离群点
        filtered_points = remove_outliers(blocks_infos[i]["pcd"].points)
        # block_pcd_bbox2 = [np.min(blocks_infos[i]["pcd"].points,axis=0),np.max(blocks_infos[i]["pcd"].points,axis=0)]
        block_pcd_bbox = [np.min(filtered_points,axis=0),np.max(filtered_points,axis=0)]
        # print(block_pcd_bbox2,block_pcd_bbox)
        
        block_pcd_bbox = [
            [block_pcd_bbox[0][0],block_pcd_bbox[1][0]],
            [block_pcd_bbox[0][1],block_pcd_bbox[1][1]],
            [block_pcd_bbox[0][2],block_pcd_bbox[1][2]]
        ]
        block_pcd_bbox_points = []
        for offect in offsets:
            block_pcd_bbox_points.append([block_pcd_bbox[0][offect[0]],block_pcd_bbox[1][offect[1]],block_pcd_bbox[2][offect[2]]])
        block_pcd_bbox_points_np = np.asarray(block_pcd_bbox_points)#8,3
        block_pcd_bbox_points_np = np.concatenate((block_pcd_bbox_points_np,np.ones_like(block_pcd_bbox_points_np[:,0][...,None])),axis=-1).T#齐次化,4,8
        
        ndc_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ndc_polygon = Polygon(ndc_coords)
        
        sixe_plane_index = [
            [0,2,3,1],
            [1,3,7,5],
            [5,7,6,4],
            [6,2,0,4],
            [2,3,7,6],
            [0,1,5,4]
        ]
        
        for j in range(len(cam_centers)):
            image_name = cam_infos[j].image_name
            if image_name in blocks_infos[i]["image_names"]:#has added
                continue
            total_area = 4#总面积为4
            
            #计算bbox在image上的投影坐标
            w2c = cam_w2cs[j]#4,4,view矩阵
            proj_mat = getProjectionMatrix(near,far,cam_infos[j].FovX,cam_infos[j].FovY)#4 * 4,proj矩阵
            
            #3d gaussian
            vp = proj_mat @ w2c
            block_pcd_bbox_points_np_proj = vp @ block_pcd_bbox_points_np#4,8
            p_w = 1.0 / (block_pcd_bbox_points_np_proj[3,:] + 0.0000001);
            p_proj = (block_pcd_bbox_points_np_proj[:3,:] * p_w[None,:]).T #8,3 xyz
            p_proj = p_proj[...,:2]
            
            #计算p_proj的凸包
            hull = ConvexHull(p_proj)
            hull_vertices_coords = p_proj[hull.vertices]
            
            #求取相交面积
            proj_polygon2 = Polygon(hull_vertices_coords)
            intersection_area = ndc_polygon.intersection(proj_polygon2).area
            
            if intersection_area / total_area > visibility_threshold:    
                flag = True
                for plane_index in sixe_plane_index:
                    plane_coord = p_proj[plane_index]#4 2
                    if not is_convex_polygon(plane_coord):
                        flag = False
                        break
                if flag:
                    blocks_infos[i]["image_names"].append(image_name)
                    blocks_infos[i]["add_images"].append(image_name)
                    blocks_infos[i]["cam_indexs"].append(j)
                    
                    
                if flag:  
                    for plane_index in sixe_plane_index:
                        plane_coord = p_proj[plane_index]
                        plane_coord_x = plane_coord[...,0]
                        plane_coord_x = np.concatenate([plane_coord_x,plane_coord_x[0][None]])
                        plane_coord_y =plane_coord[...,1]
                        plane_coord_y =  np.concatenate([plane_coord_y,plane_coord_y[0][None]])
                        plt.plot(plane_coord_x, plane_coord_y, 'b-')
                    
                    x = [-1, 1, 1, -1, -1]
                    y = [-1, -1, 1, 1, -1]

                    # 绘制正方形
                    plt.plot(x, y, 'b-')
                    
                    plt.savefig(os.path.join(colmap_data_path,"test.png"))
                    plt.clf()
            else:
                flag = True
                for plane_index in sixe_plane_index:
                    plane_coord = p_proj[plane_index]#4 2
                    if not is_convex_polygon(plane_coord):
                        flag = False
                        break
                if flag:  
                    for plane_index in sixe_plane_index:
                        plane_coord = p_proj[plane_index]
                        plane_coord_x = plane_coord[...,0]
                        plane_coord_x = np.concatenate([plane_coord_x,plane_coord_x[0][None]])
                        plane_coord_y =plane_coord[...,1]
                        plane_coord_y =  np.concatenate([plane_coord_y,plane_coord_y[0][None]])
                        plt.plot(plane_coord_x, plane_coord_y, 'b-')
                    
                    x = [-1, 1, 1, -1, -1]
                    y = [-1, -1, 1, 1, -1]

                    # 绘制正方形
                    plt.plot(x, y, 'b-')
                    
                    plt.savefig(os.path.join(colmap_data_path,"test.png"))
                    plt.clf()
    
    for i in range(block_num):
        print("block",i,"_add:"+str(len(blocks_infos[i]["add_images"])) + " views")
    
    #Coverage-based
    for i in range(block_num):
        for j in tqdm(range(len(blocks_infos[i]["cam_indexs"]))):
            pcd_indexs = blocks_infos[i]["pcd_indexs"]
            pcd_indexs = np.asarray(pcd_indexs)
            cam_index = blocks_infos[i]["cam_indexs"][j]
            w2c = cam_w2cs[cam_index]
            proj_mat = getProjectionMatrix(near,far,cam_infos[cam_index].FovX,cam_infos[cam_index].FovY)#4 * 4,proj矩阵
            vp = proj_mat @ w2c
            # for k in tqdm(range(pcd_num)):
            #     if k in pcd_indexs:
            #         continue #已经添加
            #     pcd_point = pcd_points[k].reshape(3,1)
            #     pcd_point = np.concatenate([pcd_point,np.ones((1,1))],axis=0)#4,1
            #     #判断点云是否在视锥内部
            #     # vp = proj_mat @ w2c
            #     pcd_point_proj = (vp @ pcd_point).reshape(4,1)#4,1
            #     p_w = 1.0 / (pcd_point_proj[3] + 0.0000001);
            #     p_proj = (pcd_point_proj[:3] * p_w[None,:]).T#3,1
            #     p_proj = p_proj.reshape(3)
            #     if p_proj[0] > -1.0 and p_proj[0] < 1.0 and p_proj[1] > -1.0 and p_proj[1] < 1.0 and p_proj[2] > 0.0 and p_proj[2] < 1.0:#valid
            #         blocks_infos[i]["pcd_points"].append(pcd_points[k])
            #         blocks_infos[i]["pcd_colors"].append(pcd_colors[k] * 255)
            #         blocks_infos[i]["pcd_normals"].append(pcd_normals[k])
            #         blocks_infos[i]["pcd_indexs"].append(k)
            
            pcd_points_t = pcd_points.T#3,n
            pcd_points_t = np.concatenate([pcd_points_t,np.ones_like(pcd_points_t[0,...][None,...])],axis=0)#4,n
            pcd_points_t_proj = vp @ pcd_points_t
            p_w = 1.0 / (pcd_points_t_proj[3,:] + 0.0000001);
            p_proj = (pcd_points_t_proj[:3,:] * p_w[None,:]).T #n,3 xyz
            p_proj_mask = (p_proj[...,0] < 1.0) * (p_proj[...,0] > -1.0) * (p_proj[...,1] < 1.0) * (p_proj[...,1] > -1.0)  * (p_proj[...,2] < 1.0) * (p_proj[...,2] > 0.0) 
            valid_index = np.where(p_proj_mask)[0]
            indices_to_delete = np.in1d(valid_index,pcd_indexs,assume_unique=True)
            valid_index_filter = np.delete(valid_index, np.where(indices_to_delete))
            
            blocks_infos[i]["pcd_indexs"] = np.concatenate([valid_index_filter,pcd_indexs])
            blocks_infos[i]["pcd_points"] = np.concatenate([blocks_infos[i]["pcd_points"],pcd_points[valid_index_filter]])
            blocks_infos[i]["pcd_colors"] = np.concatenate([blocks_infos[i]["pcd_colors"],pcd_colors[valid_index_filter] * 255])
            blocks_infos[i]["pcd_normals"] = np.concatenate([blocks_infos[i]["pcd_normals"],pcd_normals[valid_index_filter]])
            
        block_pcd_points = np.stack(blocks_infos[i]["pcd_points"],axis=0)
        block_pcd_colors = np.stack(blocks_infos[i]["pcd_colors"],axis=0)
        block_pcd_normals = np.stack(blocks_infos[i]["pcd_normals"],axis=0)
        block_pcd = BasicPointCloud(points=block_pcd_points, colors=block_pcd_colors, normals=block_pcd_normals)
        
        blocks_infos[i]["pcd"] = block_pcd
        storePly(os.path.join(split_block_path,str(i),"pcd_" + str(i) + "_cov.ply"), block_pcd_points, block_pcd_colors)
        print(f"block{i} has {len(block_pcd_points)} points,contain {len(block_pcd_points)/len(pcd_points)}")   
        
    #保存block_info
    block_infos_sample = []
    for block_info in blocks_infos:
        block_infos_sample.append(
            {
                "block_id":block_info["block_id"],
                "orgin_block_range":block_info["orgin_block_range"],
                "extend_block_range":block_info["extend_block_range"],
                "image_names":block_info["image_names"],
                "add_images":block_info["add_images"],
            }
        )
    with open(os.path.join(split_block_path,"blocks_info.json"), 'w') as f:
        json.dump(block_infos_sample, f)
    print("done!")
if __name__ == "__main__": 
    colmap_data_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js"
    invalid_image_names = []
    
    x_block_num = 2
    z_block_num = 2
    val_image_path = "/home/chenyi/nerf-frame/data/mega/rubble-pixsfm/val/rgbs"
    val_name = [path.split(".")[0] for path in os.listdir(val_image_path)]
    bound_extend = 0.2#边界扩张
    visibility_threshold = 0.25 #能见度阈值
    
    near = 0.01
    far = 100.0
    
    
    scence_block(colmap_data_path,x_block_num,z_block_num,val_name,bound_extend,visibility_threshold,near,far,invalid_image_names)