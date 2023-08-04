#!/usr/bin/env bash

local="/media/trunk/sata/bag/VTBJ/VTI-"

help() {
    echo -e "usage:\nossd 8611(local_dir) <remote-oss-path>"
}

path() {
    #input: local_path
    local=$1
    if [ -d "${local}" ]; then
        echo "${local} exists."
    else
        echo "local path: ${local}"
        mkdir -p ${local}
    fi
}

download() {
    #input: local_path, remote_path
    local=$1
    remote=$2
    echo -e "remote path: ${remote}"
    ossutil64 cp -r ${remote} ${local} --include "man*" --include "*common*" --include "target_fusion*.log"
}

if [ $# -eq 2 ];then
    local_path=${local}$1
    path ${local_path}
    download ${local_path} $2
else
    help
fi
