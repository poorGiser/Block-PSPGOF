for ((i=0; i<8;i++)); 
do
    cmd_line="python metrics.py  -m /home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/rubble_show2/block$i --resolution 4" 
    echo "current training: $i"
    echo "$cmd_line"
    eval $cmd_line
done