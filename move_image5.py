#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path


def move_image5(in_dir: Path, dir_name: str, out_name: str):
    paths = [x for x in in_dir.iterdir() if x.is_dir()
             and (x / dir_name).exists()]  # input dir/paths/image_5
    for p in paths:
        os.system("ln -s " + str(p/dir_name) + " " + str(p/out_name))


def parse_arguments(argv):
    """parse arguments

    Args:
        argv (list): sys.argv[1:]

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, help="指定输入路径",
                        default="/media/trunk/sata/datasets/TRUNK/0802")
    parser.add_argument('--dir', type=str, help="要修改的文件夹名字",
                        default="image_5")
    parser.add_argument('--out', type=Path, help="输出的链接文件夹名字",
                        default="camera_f60")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    move_image5(args.path, args.dir, args.out)
