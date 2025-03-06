from PIL import Image
single_width = 1063
full_width = 2244
output = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/temp/Figures/adjust_width"
def adjust_tif_width(input_path, output_path, new_width):
    # 打开TIFF图片
    with Image.open(input_path) as img:
        # 获取当前宽高比
        aspect_ratio = img.height / img.width
        
        # 计算新的高度
        new_height = int(new_width * aspect_ratio)
        
        # 调整图片大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 保存调整后的图片
        # resized_img.save(output_path, dpi=img.info['dpi'], format='TIFF')
        resized_img.save(output_path, dpi=(300,300), format='TIFF')
        

# 使用示例
adjust_tif_width('/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/temp/Figures/figure/2.tif', output + '/2.tif', new_width=full_width)
