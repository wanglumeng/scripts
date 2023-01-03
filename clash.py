#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from pathlib import Path


def parse_arguments(argv):
    """parse arguments

    Args:
        argv (list): sys.argv[1:]

    Returns:
        args: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=str,
                        help="clash版本号 eg: 0.20.12")
    parser.add_argument('--path', type=Path, help="clash下载安装路径",
                        default=Path("/media/trunk/sata/download/clash"))
    return parser.parse_args(argv)


def tar_mv(args) -> None:
    """system shell commands

    Args:
        args (dict): arguments
    """
    tar_name: str = "Clash.for.Windows-" + args.version + "-x64-linux.tar.gz"
    tar_file: Path = args.path / tar_name
    out_path: Path = args.path / Path("clash-" + args.version)
    temp_out: Path = args.path / \
        Path("Clash for Windows-" + args.version + "-x64-linux")

    # out_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_path, ignore_errors=True)

    os.system("tar -xvf " + str(tar_file) + " -C " + str(args.path))

    os.rename(temp_out, out_path)


def zsh_aliases(args):
    aliases = os.path.expanduser("~/.zsh_aliases")
    with open(aliases, 'r+') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line.find("alias clash") != -1:
                default = 'alias clash="/media/trunk/sata/download/clash/clash-0.20.11/cfw &"'
                lines[index] = default.replace("0.20.11", args.version)
                f.seek(0)
                f.writelines(lines)
                return


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    tar_mv(args)
    zsh_aliases(args)
