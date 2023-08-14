import os
import rosbag
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='rosbag reader')
    parser.add_argument(
        '--bags_dir', help='bags directory to be read', default='/media/trunk/sata/bag/hota_0613/scenes/common_bag/rain_old')
    parser.add_argument(
        '--out_dir', help='bags output directory', default='/media/trunk/sata/bag/hota_0613/scenes/common_bag/rain')
    # parser.add_argument(
    #     '--topic_name', help='topic to be read', default='/perception/objects')
    # parser.add_argument(
    #     '--msg_type', help='msg type to be read', default='perception_msgs/Objects')
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    if os.path.exists(args.bags_dir)== False:
        print("bags_dir not exist!")
        return
    if os.path.exists(args.out_dir)== False:
        os.makedirs(args.out_dir)
        
    for filename in os.listdir(args.bags_dir):
        if filename.endswith(".bag"):
            print("parsing rosbag {} ...".format(filename))
            bag = rosbag.Bag(os.path.join(args.bags_dir, filename), 'r')
            output_bag = rosbag.Bag(os.path.join(args.out_dir, filename), 'w')
    
            for topic, msg, t in bag.read_messages():
                # 在这里对消息进行修改
                if topic == '/ARS430_input':
                    for p_object in msg.Data:
                        p_object.ProbOfObs = p_object.ProbOfObs * 100

                # 将修改后的消息写入新的ROS bag文件
                output_bag.write(topic, msg, t)

            # 关闭ROS bag文件
            bag.close()

            # 关闭输出的ROS bag文件
            output_bag.close()
            print("Done Output: {}".format(os.path.join(args.out_dir, filename)))      
        else:
            continue
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    main()
