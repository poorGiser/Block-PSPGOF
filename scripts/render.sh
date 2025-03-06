for ((i=0; i<8;i++)); 
do
    cmd_line="python render.py -block_index $i -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rubble -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble_show2/block$i -resolution 4 --skip_train" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done