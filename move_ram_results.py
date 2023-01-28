#!/usr/bin/env python
import os
import sys
import shutil
import argparse
from pathlib import Path


def move_results(base: Path, in_path: Path):
    for f in (base/in_path).glob("*.txt"):
        out = base/f.stem/"RAM"/"data"
        os.makedirs(out, exist_ok=True)
        shutil.copy(f, out/"tracking_result_eval.txt")
        ...


def parse_arguments(argv):
    """parse arguments

    Args:
        argv (list): sys.argv[1:]

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, help="base path",
                        default="/media/trunk/sata/datasets/TRUNK/0802")
    parser.add_argument('--input', type=str, help="input path",
                        default="results_trunk_mot/trunk")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    move_results(args.path, args.input)
