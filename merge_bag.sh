#!/usr/bin/env bash

dir1="/home/nvidia/data/ssd/wlm/bag/"
dir2="/home/nvidia/data/ssd/wlm/bag/only_dr"
out="/home/nvidia/data/ssd/wlm/bag/mergebags/"

group=("common" "rain")

for g in ${group[*]}; do
    for i in $(ls ${dir1}/$g); do
        echo $i
        rosbag record -a -O ${out}/$g/$i &
        rosbag play --clock ${dir1}/$g/$i ${dir2}/$g/$i
        rosnode kill `rosnode list | grep "/record_*"`
    done
done
