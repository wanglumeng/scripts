#!/usr/bin/env bash

# usage:
# modify in_dir & out_dir
# ./ run it

# 用途：
# delete topic: /pnc_msgs/vehicle_state

in_dir="/home/nvidia/data/ssd/zyq/work/metricslidarperception/bag_data_at128_v4/rain/"
out_dir="/home/nvidia/data/ssd/wlm/bag/rain/"

function filter() {
    # input params:
    # 1. input dir
    # 2. output dir
    # 3. filename
    
    # 不要的topic写在下面
    
    rosbag filter $1/$3 $2/$3 \
    'topic != "/pnc_msgs/vehicle_state"' # \
    # and topic != "/hadmap_server/planning_path"'
}

# 使用if语句检查输出目录是否存在
if [ ! -d "$out_dir" ]; then
    # 如果目录不存在，使用mkdir命令创建它
    mkdir -p "$out_dir"
    echo "Directory created: $out_dir"
else
    echo "Directory already exists: $out_dir"
fi

for file in "$in_dir"*
do
    if [ -f "$file" ]; then
    filename=$(basename $file)
    filter $in_dir $out_dir $filename
    fi
done

