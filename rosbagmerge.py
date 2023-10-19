#!/usr/bin/env python3
import os
import rosbag

dir1="/home/nvidia/data/ssd/wlm/bag/"
dir2="/home/nvidia/data/ssd/wlm/bag/only_dr"
out="/home/nvidia/data/ssd/wlm/bag/mergebags/"

group=["common", "rain"]

def merge(inbags:list, outbag:str):
    """
    inbags:输入要合并的bag文件列表
    outbag:输出bag文件名
    """
    # 创建一个新的bag文件用于合并
    merged_bag = rosbag.Bag(outbag, 'w')
    
    # 遍历每个输入bag文件并将其写入到合并的bag文件中
    for bag_file in inbags:
        with rosbag.Bag(bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                merged_bag.write(topic, msg, t)
    # 关闭合并的bag文件
    merged_bag.close()
    
    print("合并完成")

if __name__ == '__main__':
    # merge(['/home/nvidia/data/ssd/wlm/bag/common/scene01_1_2022-08-02-14-51-47_5.bag', '/home/nvidia/data/ssd/wlm/bag/only_dr/common/scene01_1_2022-08-02-14-51-47_5.bag'], '/home/nvidia/data/ssd/wlm/bag/mergebags/common/scene01_1_2022-08-02-14-51-47_5.bag')
    for g in group:
        print(g)
        for i in sorted(os.listdir(os.path.join(dir1,g))):
            print(i)
            inbags=[os.path.join(dir1, g, i), os.path.join(dir2, g, i)]
            outbag=os.path.join(out, g, i)
            merge(inbags, outbag)
