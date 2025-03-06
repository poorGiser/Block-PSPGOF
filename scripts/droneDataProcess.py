'''
对无人机数据做处理
'''
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
def droneDataProcess(root_dir,save_dir,val_image_num):
    image_path = os.path.join(root_dir,"undistortionX0Y0")
    image_names = os.listdir(image_path)
    IMG_POS_Path = os.path.join(root_dir,"resultEO.txt")
    
    save_image_path = os.path.join(save_dir,"input")
    os.makedirs(save_image_path,exist_ok=True)
    i = 0
    base_names = []
    geo_registration_text_file = []
    geo_registration_text_dict = {}
    
    name_xyz_dict = {}
    df = pd.read_csv(IMG_POS_Path, sep='\t', header=None).values
    for row in df:
        name_xyz_dict[row[0]] = row[1:]
    
    camera_centers = []
    
    for image_name in tqdm(image_names):
        base_name = image_name.split('.')[0]
        image_data = name_xyz_dict.get(base_name)
        if image_data is None:
            print(base_name)
            continue
            
        
        if len(image_data) == 6:
            X = float(image_data[0])
            Y = float(image_data[1])
            Z = float(image_data[2])
            
            #将YZ互换并对y轴取反
            temp = Y
            Y = -Z
            Z = temp
            
            camera_centers.append([X,Y,Z])
            
            #将图片复制到save_dir
            image_base_name = f"{i:04d}"
            # shutil.copy(os.path.join(image_path,image_name),save_image_path + "/" + image_base_name + ".jpg")
            # geo_registration_text_file.append([image_base_name + ".jpg",X,Y,Z])
            geo_registration_text_dict[image_base_name] = [X,Y,Z]
            base_names.append(image_base_name)
            i+=1
        else:
            image_base_name = f"{i:04d}"
            print(image_base_name + ".jpg")
            i+=1
    
    
    camera_centers_np = np.asarray(camera_centers)
    camera_centers_plane = camera_centers_np[:,[0,2]]
    camera_center_heights = camera_centers_np[:,1]
    #计算相机中心
    camera_centers_mean = np.mean(camera_centers_plane,axis=0)
    
    #平移参数
    translations = np.asarray([-camera_centers_mean[0],-np.min(camera_center_heights) / 2,-camera_centers_mean[1]])
    
    #缩放参数
    expect_scale = 20
    x_span = (np.max(camera_centers_plane[:,0]) - np.min(camera_centers_plane[:,0]))
    y_span = (0 - np.min(camera_centers_np[:,1]))
    z_span = (np.max(camera_centers_plane[:,1]) - np.min(camera_centers_plane[:,1]))
    
    max_span = max((x_span, y_span,z_span))
    scale = expect_scale / max_span
    
    
    scales = np.asarray([scale,scale,scale])
    
    for image_base_name in base_names:
        xyz = np.asarray(geo_registration_text_dict[image_base_name])
        #归一化
        normalized_xyz = (xyz + translations) * scales
        geo_registration_text_file.append([image_base_name + ".jpg",normalized_xyz[0],normalized_xyz[1],normalized_xyz[2]])
        
    all_image_num = len(base_names)
    #随机生成val_image_num个随机数，范围为0-all_image_num-1
    val_image_indexs = np.random.choice(all_image_num,val_image_num,replace=False)
    
    val_image_base_names = []
    for val_image_index in val_image_indexs:
        val_image_base_names.append(base_names[val_image_index])
        
    output_file = os.path.join(save_dir,"geo_registration.txt")
    array = np.asarray(geo_registration_text_file)
    with open(output_file, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)) + '\n')
    #写入json信息             
    val_image_json_path = os.path.join(save_dir,"val_images.json")
    with open(val_image_json_path,"w") as file:
        json.dump(val_image_base_names,file,indent=4)   
        
    #保存translations和scales
    transform = {}
    translations = translations.tolist()
    scales = scales.tolist()
    transform["translations"] = translations
    transform["scales"] = scales
    transformation_json_path = os.path.join(save_dir,"transform.json")
    #保存
    with open(transformation_json_path,"w") as file:
        json.dump(transform,file,indent=4)
    
    print("Done!") 

if __name__ == "__main__":
    root_dir = "/home/chenyi/data/Rural"
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural2"
    val_image_num = 20
    droneDataProcess(root_dir,save_dir,val_image_num)
    # root_dir = "/home/chenyi/data/Rural"
    # save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural"
    # val_image_num = 20
    # droneDataProcess(root_dir,save_dir,val_image_num)
    