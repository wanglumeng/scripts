#!/usr/bin/env bash

path="common"

cd /home/nvidia/data/ssd/wlm/bag/
for bag in "${path}"/*;do
        # rosbag play common/scene01_1_2022-08-02-14-51-47_5.bag only_dr/common/scene01_1_2022-08-02-14-51-47_5.bag perception_objs_at128_v4/common/scene01_1_2022-08-02-14-51-47_5.bag vision_output/front/bag/common/scene01_1_2022-08-02-14-51-47_5.bag vision_output/side/bag/common/scene01_1_2022-08-02-14-51-47_5.bag
        rosbag play --clock $bag only_dr/$bag perception_objs_at128_v4/$bag vision_output/front/bag/$bag vision_output/side/bag/$bag

done
