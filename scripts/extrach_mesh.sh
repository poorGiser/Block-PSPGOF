for ((i=2; i<3;i++)); 
do
    cmd_line="CUDA_VISIBLE_DEVICES=1 python extract_mesh.py -block_index $i -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/js -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/js/block$i -resolution 4 --iteration 30000 --texture_mesh --filter_mesh --plane_iso 2" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done