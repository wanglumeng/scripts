#!/usr/bin/env bash

# 指定文件路径
dir="/home/trunk/work/code/tad_soc_release/src/perception/target_fusion/params"

# 匹配文件列表
files="${dir}/sensor_config_*.toml"

# 遍历匹配文件并重命名
for file in $files; do
    if [ -f "$file" ]; then
        new_file="${file//sensor_config_/ros_sensor_config_}"  # 替换文件名
        mv "$file" "$new_file"
        echo "已将 $file 重命名为 $new_file"
    fi
done
