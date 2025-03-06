#!/bin/bash

# 定义字符串数组
# string_array=("power_station" "rural" "building" "rubble" "campus" "residence")
string_array=("power_station")

# 定义固定字符串
fixed_string="colmap"

# 循环遍历数组
for value in "${string_array[@]}"; do
    # 拼接命令
    command1="$fixed_string patch_match_stereo --workspace_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value --workspace_format COLMAP --PatchMatchStereo.geom_consistency true PatchMatchStereo.gpu_index=0,1 PatchMatchStereo.max_image_size=1361"
    echo $command1
    eval $command1

    command2="$fixed_string stereo_fusion --workspace_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value --workspace_format COLMAP --input_type geometric --output_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value/fused.ply"
    echo $command2
    eval $command2
    
    command3="$fixed_string poisson_mesher --input_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value/fused.ply --output_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value/meshed-poisson.ply"
    echo $command3
    eval $command3

    command4="$fixed_string delaunay_mesher --input_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value --output_path /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/$value/meshed-delaunay.ply"
    echo $command4
    eval $command4
done
