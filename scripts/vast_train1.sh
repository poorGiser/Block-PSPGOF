#rubble
# for ((i=2; i<8;i++)); 
# do
#     cmd_line="python train.py -block_index $i -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble_show2/block$i -resolution 4" 
#     echo "current training: $i"
#     echo "$cmd_line"
#     eval $cmd_line
# done

#residence
for ((i=4; i<8;i++)); 
do
    cmd_line="CUDA_VISIBLE_DEVICES=1 python train.py -block_index $i -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/campus -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/campus/block$i -resolution 4 -gpu_count 1" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done