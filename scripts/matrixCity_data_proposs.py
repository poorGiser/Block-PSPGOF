#preproocess for matrixcity dataset
import os
import json
# import re
import shutil
import numpy as np
from tqdm import tqdm
def format_to_four_digit_string(value):
    # 使用字符串格式化将整数值格式化为四位字符串
    return f"{value:04d}"
def matrixCity_data_proposs(root_dir,save_dir):
    scene_index = "block_all"
    splits = ["train","test"]
    
    scene_dir = os.path.join(save_dir,f"matrix_city_all")
    os.makedirs(scene_dir,exist_ok=True)
    
    image_save_dir = os.path.join(scene_dir,"input")
    os.makedirs(image_save_dir,exist_ok=True)
    
    i = 0
    
    geo_registration_text_file = []
    val_basenames = []
    for split in splits:
        image_root_dir = os.path.join(root_dir,split)
        json_path = os.path.join(root_dir,"pose",f"block_all",f"transforms_{split}.json")
        if os.path.join(json_path):
            with open(json_path) as file:
                params = json.load(file)
                frames = params["frames"]
                for frame in tqdm(frames):
                    transform_matrix = frame["transform_matrix"]
                    file_path = frame["file_path"]
                    relative_path = os.path.relpath(file_path, f"../../{split}")
                    filename = os.path.basename(file_path)
                    
                    file_basename = filename.split(".")[0]
                    
                    image_path = os.path.join(image_root_dir,relative_path)
                    
                    new_filename = format_to_four_digit_string(i) + ".png"
                    new_basename = new_filename.split(".")[0]
                    
                    if os.path.exists(image_path):
                        dst_file = os.path.join(image_save_dir,new_filename)
                        shutil.copy(image_path, dst_file)
                        
                    X = transform_matrix[0][3]
                    Y = transform_matrix[1][3]
                    Z = transform_matrix[2][3]
                    
                    
                    geo_registration_text_file.append([new_filename,X,Y,Z])
                    if split == "test":
                        val_basenames.append(new_basename)
                    i+=1
    #写入geo_registration_text_file .txt文件
    output_file = os.path.join(scene_dir,"geo_registration.txt")
    array = np.asarray(geo_registration_text_file)
    with open(output_file, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)) + '\n')
    #写入json信息             
    val_image_json_path = os.path.join(scene_dir,"val_images.json")
    with open(val_image_json_path,"w") as file:
        json.dump(val_basenames,file,indent=4)   
    print("Done!") 
                    
if __name__ == "__main__":
    root_dir = "/home/chenyi/data/maxcity/aerial"
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data"
    matrixCity_data_proposs(root_dir=root_dir,save_dir=save_dir)
    
    