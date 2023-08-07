#!/usr/bin/env python3

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

suffix = "*.bag"
scenes_0802 = ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1",
               "07_1", "08_1", "09_1", "10_1", "11_1", "12_1", "13_1"]
scenes_rain = ["13-29", "13-30", "13-34", "11-36", "13-41"]

def parse_arguments(argv):
    """parse arguments

    Args:
        argv (list): sys.argv[1:]

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help="指定某个场景eg.01_1")
    parser.add_argument('--fusion_ws', type=Path, help="传感器融合代码workspace",
                        default=os.path.expanduser(
                            "~/work/code/tad_soc_release/"))
    parser.add_argument('--output', type=Path, help="输出路径metrics仓库里的data/fusion",
                        default=os.path.expanduser(
                            "~/work/code/metricslidarperception/data/fusion"))
    parser.add_argument('--bags', type=Path, help="除了lidar目标其他topic的bag路径",
                        default="/media/trunk/sata/bag/hota_0613/bag_data_without_obj")
    parser.add_argument('--objs', type=Path, help="lidar目标bag路径",
                        default="/media/trunk/sata/bag/hota_0613/perception_objs")
    parser.add_argument('--vision', type=Path, help="vision bag路径",
                        default="/media/trunk/sata/bag/hota_0613/vision_objs")
    parser.add_argument('--vision_side', type=Path, help="vision_side bag路径",
                        default="/media/trunk/sata/bag/hota_0613/vision_side_objs")

    return parser.parse_args(argv)


def xreplace(lines: list, index: int, str_: str, x: float = 0):
    """replace lines[index] to x(when str_ in line)

    Args:
        lines (list): file.readlines
        index (int): line index
        str_ (str): string to find
        x (float, optional): new value. Defaults to 0.

    Returns:
        bool: True for modified
    """
    if str_ in lines[index]:
        l2 = "{} {} {}\n".format(lines[index].split(
            ' ')[0], lines[index].split(' ')[1], x)
        lines[index] = l2
    else:
        return False

    return True


def calibration(file: Path):
    """modify calibration parameters

    Args:
        file (Path): calibration.cpp. 

    Raises:
        Exception: path error
    """
    if file.is_file():
        with open(file, 'r+') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                xreplace(lines, index, "#define TRANSLATION_X", 5.36)
                xreplace(lines, index, "#define TRANSLATION_Y", 0.0)
                xreplace(lines, index, "#define TRANSLATION_Z", 0.65)
                xreplace(lines, index, "#define ROTATION_ROLL", 0.0)
                xreplace(lines, index, "#define ROTATION_PITCH", 0.0)
                xreplace(lines, index, "#define ROTATION_YAW", 1.0)
            f.seek(0)
            f.writelines(lines)
    else:
        raise Exception(f"calibration.cpp file error!\n{file}")


def build(fusion_ws: Path):
    """build after modifying calibration paramters
y
    Args:
        fusion_ws (Path): source code
    """
    # calibration(
    #     fusion_ws/"src/perception/target_fusion/src/sensors/calibration/calibration.cpp")
    os.chdir(fusion_ws)
    subprocess.check_call("catkin clean -y", shell=True)
    # subprocess.check_call("catkin install", shell=True)
    subprocess.check_call("catkin build target_fusion", shell=True)
    os.chdir(os.curdir)


def launch(args, bag_scene: str = "01_1", launch_flag: bool = True):
    """launch target_fusion, play data_bag+obj_bag

    Args:
        args (_type_): runtime arguments(include bags for bag_path, objs for obj_bag_path)
        bag_scene (str, optional): data(without lidar obj) scene number. Defaults to "01_1".
        launch_flag (bool, optional): if launch target_fusion node. Defaults to True.
    """
    bag: str = str(args.bags) + "/scene" + bag_scene + suffix
    obj: str = str(args.objs) + "/scene" + bag_scene + suffix
    vision: str = str(args.vision) + "/scene" + bag_scene + suffix
    vision_side: str = str(args.vision_side) + "/scene" + bag_scene + suffix
    bag_play: str = "rosbag play --clock " + bag + " " + \
        obj + " " + vision + " " + vision_side + """ --topics /ARS430_input /hadmap_server/current_region \
            /perception/odometry /pnc_msgs/vehicle_info2 /tf \
            /LFCr5tpRadarMsg /LRCr5tpRadarMsg /RFCr5tpRadarMsg /RRCr5tpRadarMsg \
            /clock /perception/objects /vision_lanes /vision_objects /vision_f30_objects \
            /visual_side_perception_results/left /visual_side_perception_results/right \
            /hadmap_server/current_region /hadmap_server/local_map /pnc_msgs/vehicle_state"""
    if launch_flag:
        subprocess.check_call(
            "roslaunch target_fusion sensor_fusion_offline.launch &", shell=True)
        time.sleep(12)
        subprocess.check_call(bag_play, shell=True)

    else:
        subprocess.check_call(bag_play, shell=True)


def output(sensorfusion_dir: Path, output_dir: Path) -> bool:
    """assign output_dir in source code path config toml, clean output_dir

    Args:
        sensorfusion_dir (Path): source code path
        output_dir (Path): output path (*.json)

    Returns:
        bool: True for has *.json(rm *.json)
    """
    toml = sensorfusion_dir / "src/perception/target_fusion/params/track_config.toml"
    with open(toml, 'r+') as file:
        lines = file.readlines()
        for index, line in enumerate(lines):
            if "logOn" in line:
                lines[index] = "logOn = true\n"
            elif "logPath" in line:
                lines[index] = "logPath = \"" + str(output_dir) + "\"\n"
        file.seek(0)
        file.writelines(lines)
    source = "source "+str(sensorfusion_dir)+"/devel/setup.zsh"
    # subprocess.check_call(source, shell=True)
    rm = "rm -r "+str(output_dir)+"/*.json"
    print(os.listdir(output_dir))
    output_list = os.listdir(output_dir)
    for s in output_list:
        if ".json" in s:
            subprocess.check_call(rm, shell=True)  # rm temp output dir(*.json)
            return True
    return False


def save(args, scene: str = "1_1", type: str = "json"):
    """save json files to output path

    Args:
        args (dict): arguments
        scene (str, optional): scene_name. Defaults to "1_1".
        type (str, optional): json or txt. Defaults to "json".
        TODO: txt
    """
    bag_path = list(args.bags.glob("*scene"+scene+"*.bag"))
    out_dir: Path = args.output / bag_path[0].name
    print(args.output / bag_path[0].name)
    subprocess.check_call("rosnode kill /target_fusion", shell=True)
    time.sleep(1)
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for json_file in args.output.glob("1*."+type):
        shutil.copy2(json_file, out_dir)
        os.remove(json_file)

def load_topics(args):
    import time
    subprocess.check_call("roscore&", shell=True)
    time.sleep(5)
    subprocess.check_call("ls " + str(args.fusion_ws) + "/entry/config/*.yaml | xargs -I @ rosparam load @", shell=True)
    subprocess.check_call("rosparam set /function/log/level 4", shell=True) #error level

def hmi(args):
    subprocess.check_call("roslaunch hmi hmi_horizontal.launch", shell=True)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    load_topics(args)
    #hmi(args)
    if args.scene:  # play 1 scene
        launch(args, args.scene, False)  # only play bag
    else:
        # build(args.fusion_ws)
        output(args.fusion_ws, args.output)
        for scene in scenes_0802:
            subprocess.check_call("clear", shell=True)
            launch(args, scene)
            save(args, scene)
