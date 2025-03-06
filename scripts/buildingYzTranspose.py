'''
building 数据集 colmap曼哈顿对齐后不是y轴向下,需要转换
'''
import os 
import numpy as np
import torch
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras,storePly,fetchPly
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from tqdm import tqdm
from scipy.spatial.transform import Rotation
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-8)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-8)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-8)
    return qw, qx, qy, qz

def rot2quaternion(rotation_matrix):
    r3 = Rotation.from_matrix(rotation_matrix)
    qua = r3.as_quat()
    return qua 
def rotation_matrix_to_quaternion2(R,image_id):
    # 确保输入是一个 3x3 的旋转矩阵
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix."
    is_invalid = False
    # 计算迹
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    # 计算四元数
    s = np.sqrt(trace + 1.0) * 2  # 4 * w
    w = 0.25 * s
    x = (R[2, 1] - R[1, 2]) / s
    y = (R[0, 2] - R[2, 0]) / s
    z = (R[1, 0] - R[0, 1]) / s
    # if trace > 0:
    #     s = np.sqrt(trace + 1.0) * 2  # 4 * w
    #     w = 0.25 * s
    #     x = (R[2, 1] - R[1, 2]) / s
    #     y = (R[0, 2] - R[2, 0]) / s
    #     z = (R[1, 0] - R[0, 1]) / s
    # else:
    #     is_invalid = True
    #     if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    #         s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # 4 * x
    #         w = (R[2, 1] - R[1, 2]) / s
    #         x = 0.25 * s
    #         y = (R[0, 1] + R[1, 0]) / s
    #         z = (R[0, 2] + R[2, 0]) / s
    #     elif R[1, 1] > R[2, 2]:
    #         s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # 4 * y
    #         w = (R[0, 2] - R[2, 0]) / s
    #         x = (R[0, 1] + R[1, 0]) / s
    #         y = 0.25 * s
    #         z = (R[1, 2] + R[2, 1]) / s
    #     else:
    #         s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # 4 * z
    #         w = (R[1, 0] - R[0, 1]) / s
    #         x = (R[0, 2] + R[2, 0]) / s
    #         y = (R[1, 2] + R[2, 1]) / s
    #         z = 0.25 * s
    
    return np.array([w, x, y, z]),is_invalid
    
def read_txt_file_to_list(file_path):
    # with open(file_path, 'r') as file:
        # lines = file.readlines()
        # lines = [line.strip() for line in lines]  # 去除每行末尾的换行符]
    with open(file_path, "r",encoding='utf-8') as fid:
        lines = []
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)

    return lines
# def read_txt2(path):
#     lines = []
#     with open(path, "r") as fid:
#         while True:
#             line = fid.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if len(line) > 0 and line[0] != "#":
#                 lines.append(line)

#                 # break
#     return lines

def write_list_to_txt_file(output_list, file_path):
    with open(file_path, 'w') as file:
        for i,line in enumerate(output_list):
            if i != len(output_list) - 1:
                file.write(line + '\n')
            else:
                file.write(line)
            
            
if __name__ == "__main__":
    # root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/building/txtmodel"
    # root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/matrix_city_all/txtmodel"
    root_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js/txtmodel"

    ply_path = os.path.join(root_dir, "points3D.ply")
    bin_path = os.path.join(root_dir, "points3D.bin")
    txt_path = os.path.join(root_dir, "points3D.txt")

    # if not os.path.exists(ply_path):
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
        #将ply的坐标系转换
    xyz = xyz[:,[0,2,1]]
    xyz[...,1] = -xyz[...,1]
        
    storePly(ply_path, xyz, rgb)
    
    count = 0
    point_list = read_txt_file_to_list(txt_path)
    write_points = []
    for point_line in tqdm(point_list):
        if point_line[0] == "#":
            write_points.append(point_line)
            continue
        elems = point_line.split()   
        #修改x y z
        elems[1] =  str(xyz[count][0])
        elems[2] =  str(xyz[count][1])
        elems[3] =  str(xyz[count][2])
        new_line = ' '.join(elems)
        write_points.append(new_line)
        count += 1
    new_point_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js/txtmodel/new/points3D.txt"
    write_list_to_txt_file(write_points,new_point_path)
    del point_list
    del write_points
        
    #读取相机外参
    image_info_path  = os.path.join(root_dir,"images.txt")
    image_list = read_txt_file_to_list(image_info_path)
    cam_extrinsics = read_extrinsics_text(image_info_path)
    
    tr = np.asarray([
        [1,0,0,0],
        [0,0,-1,0],
        [0,1,0,0],
        [0,0,0,1]
    ],dtype=np.float64)
    
    new_vecs = {}
    new_T_s = {}
    invalid_id = []
    all_qevcs = []
    # i = 0
    for image_ext in cam_extrinsics:
        image_id = str(cam_extrinsics[image_ext].id)
        qvec = cam_extrinsics[image_ext].qvec
        all_qevcs.append(qvec)

        #四元数转旋转矩阵
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array(cam_extrinsics[image_ext].tvec)
        # T = np.matmul((-R), T)
        
        w2c = getWorld2View2(R,T).astype(dtype=np.float64)
        # new_w2c = tr @ w2c @ np.linalg.inv(tr)
        new_w2c = w2c @ np.linalg.inv(tr)
        
        # new_c2w = np.linalg.inv(new_w2c)
        
        new_rot = new_w2c[:3,:3]
        # new_rot = np.transpose(new_rot)
        # new_vec = rotmat2qvec(new_rot)
        new_vec,is_invalid = rotation_matrix_to_quaternion2(new_rot,image_id)
        is_invalid = False
        if is_invalid:
            invalid_id.append(image_id)
        if(image_id == "754"):
            print(1)
        
        new_T = new_w2c[:3,3]
        # new_T = np.matmul((-new_rot), new_T)cc
        
        
        new_T_s[image_id] = new_T
        new_vecs[image_id] = new_vec
    print(len(invalid_id))
    write_images = []
    # count = 0
    for image_line in tqdm(image_list):
        if len(image_line[0]) == 0 or image_line[0] == "#":
            write_images.append(image_line)
            continue
        elems = image_line.split()   
        id = elems[0]
        if not id.isdigit():
            write_images.append(image_line)
            continue
        if id in invalid_id:
            write_images.append(image_line)
            continue

        #修改x y z
        # if id in new_vecs:
        elems[1] =  str(new_vecs[id][0])
        elems[2] =  str(new_vecs[id][1])
        elems[3] =  str(new_vecs[id][2])
        elems[4] =  str(new_vecs[id][3])
        elems[5] =  str(new_T_s[id][0])
        elems[6] =  str(new_T_s[id][1])
        elems[7] =  str(new_T_s[id][2])
    
        new_line = ' '.join(elems)
        write_images.append(new_line)
        # count += 1
    new_image_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js/txtmodel/new/images.txt"
    write_list_to_txt_file(write_images,new_image_path)
    print("Done!")
    
        
    
    
    
    