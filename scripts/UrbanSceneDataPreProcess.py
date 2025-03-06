'''
根据mega-nerf的mappings.txt文件,处理urban3d的数据
图片重命名、保存到同一个文件夹
'''
import os
import pandas as pd
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = "/home/chenyi/data/campus/2101-OBL-SZ-VCC-shenda-1---5881-P"
    save_dir = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/campus/input"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    photo_dir = os.path.join(root_dir,"photos")
    mapping_path = os.path.join(root_dir,"mappings.txt")
    
    df = pd.read_csv(mapping_path, sep=',',header=None)
    map_data = df.values
    
    # indexs = ["A","B","C"]
    # indexs = ["1","2","3","4"]
    indexs = ["1","2","110MEDIA","111MEDIA","115MEDIA","116MEDIA","117MEDIA","118MEDIA","119MEDIA"]
    
    
    for index in indexs:
        images_path = os.path.join(photo_dir,index)
        image_names = os.listdir(images_path)
        for image_name in tqdm(image_names):
            image_path = os.path.join(images_path,image_name)
            hz = image_name.split(".")[1]
            if hz != "JPG":
                continue
            
            #满足条件的映射
            mapping_name = None
            for i in range(len(map_data)):
                if index == map_data[i,0].split("/")[0] and image_name in map_data[i,0]:
                    mapping_name = map_data[i,1]
                    break
            if mapping_name:
                mapping_name = mapping_name.split(".")[0]
                
                shutil.copyfile(image_path, os.path.join(save_dir,mapping_name + "." + hz))
            else:
                print(image_path)
    print("Done!")
            
    
    
    