for ((i=0; i<20;i++)); 
do
    cmd_line="python render.py -block_index -1 -s /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/merge -resolution 4 -index $i --skip_train" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done