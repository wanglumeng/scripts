#!/usr/bin/env bash

in_dir="/home/nvidia/data/ssd/wlm/bag/"
out_dir="/home/nvidia/data/ssd/wlm/bag/only_dr/"

file_names_common=("scene01_1_2022-08-02-14-51-47_5.bag"
            "scene02_1_2022-08-02-15-01-18_1.bag"
            "scene03_1_2022-08-02-15-13-06_1.bag"
            "scene04_1_2022-08-02-15-14-06_2.bag"
            "scene05_1_2022-08-02-15-15-06_3.bag"
            "scene06_1_2022-08-02-15-17-06_5.bag"
            "scene07_1_2022-08-02-15-54-42_5.bag"
            "scene08_1_2022-08-02-16-11-31_2.bag"
            "scene09_1_2022-08-02-16-13-31_4.bag"
            "scene10_1_2022-08-02-17-08-07_8.bag"
            "scene11_1_2022-08-02-17-11-38_0.bag"
            "scene12_1_2022-08-02-17-41-51_2.bag"
            "scene13_1_2022-08-02-17-44-51_5.bag")
file_names_rain=("lidar_at128_2023-06-19-13-29-41_3.bag"
                 "lidar_at128_2023-06-19-13-30-41_4.bag" 
                 "lidar_at128_2023-06-19-13-34-41_8.bag" 
                 "lidar_FP1219_2023-02-09-11-36-18_8.bag"
                 "lidar_FP1219_2023-02-09-13-41-58_3.bag")

function filter() {
    # input params:
    # 1. input file
    # 2. output file
    
    # 要的topic写在下面
    
    rosbag filter $1 $2 \
    'topic == "/pnc_msgs/vehicle_state" \
    or topic == "/libfusion/dr"'
}

# 使用if语句检查输出目录是否存在
if [ ! -d "$out_dir"/common ]; then
    # 如果目录不存在，使用mkdir命令创建它
    mkdir -p "$out_dir"/common
    echo "Directory created: $out_dir"/common
else
    echo "Directory already exists: $out_dir"/common
fi
if [ ! -d "$out_dir"/rain ]; then
    # 如果目录不存在，使用mkdir命令创建它
    mkdir -p "$out_dir"/rain
    echo "Directory created: $out_dir"/rain
else
    echo "Directory already exists: $out_dir"/rain
fi

index=0
for file in "$in_dir"/20220802-new/*
do
    if [ -f "$file" ]; then
    filename=$(basename $file)
    filter $in_dir/20220802-new/$filename $out_dir/common/${file_names_common[$index]}
    index=$index+1
    fi
done

index=0
for file in "$in_dir"/20230619-new/*
do
    if [ -f "$file" ]; then
    filename=$(basename $file)
    filter $in_dir/20230619-new/$filename $out_dir/rain/${file_names_rain[$index]}
    index=$index+1
    fi
done

for file in "$in_dir"/20230209-new/*
do
    if [ -f "$file" ]; then
    filename=$(basename $file)
    filter $in_dir/20230209-new/$filename $out_dir/rain/${file_names_rain[$index]}
    index=$index+1
    fi
done
