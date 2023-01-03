#!/usr/bin/env python

import os
import sys
import json
import argparse
from pathlib import Path

objs_num = 0
ids = {}


def json2mot(json_path: Path, out_path: Path, seq: str):
    """json to mot(gt.txt)

    Args:
        json_path (Path): jsons path
        out_path (Path): output path(gt.txt)

    Returns:
        _type_: None
    """
    for j_file in sorted(json_path.glob("**/*" + seq + "/*.json")):
        with open(json_path/j_file) as fp:
            full_json = json.load(fp)
            load_objs = full_json["images"][5]["items"]  # front camera(f60)
            out_objs = ""
            if not load_objs or len(load_objs) == 0:
                return out_objs
            for obj in load_objs:
                with open(out_path/"gt.txt", "a") as out:
                    out_objs = "{},{},{},{},{},{},-1,-1,-1,-1\n".format(obj["frameNum"]+1, obj["id"], obj["box2d"][0],
                                                                        obj["box2d"][1], obj["box2d"][2]-obj["box2d"][0], obj["box2d"][3]-obj["box2d"][1])
                    out.write(out_objs)


def parse_arguments(argv):
    """parse arguments

    Args:
        argv (list): sys.argv[1:]

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, help="指定json路径",
                        default="/home/trunk/Downloads/datasets/TRUNK/0802/gt/json/json_lidar")
    parser.add_argument('--out', type=Path, help="输出路径",
                        default="/home/trunk/Downloads/datasets/TRUNK/0802/highway_2022-08-02-14-51-47.bag/gt/")
    parser.add_argument('--seq', type=str,
                        help="seq num(folder name)", default="1_100")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    json_path = args.path
    out_path = args.out
    if (out_path/"gt.txt").is_file():
        os.remove(out_path/"gt.txt")
    json2mot(json_path, out_path, args.seq)
