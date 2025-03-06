for ((i=1; i<8;i++)); 
do
    cmd_line="CUDA_VISIBLE_DEVICES=1 python extrach_mesh_lod.py -block_index $i -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block$i -resolution 4 --iteration 30000 --texture_mesh --filter_mesh --plane_iso 2" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done