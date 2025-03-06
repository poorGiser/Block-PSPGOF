for ((i=0; i<8;i++)); 
do
    cmd_line="python extract_mesh.py -block_index $i -s /home/chenyi/gaussian-splatting/data/residence -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/residense/block$i -resolution 4 --iteration 30000 --texture_mesh --filter_mesh" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done