#!/usr/bin/env bash

set -ex # -e:exit when error, -x: print

ROOT="/media/trunk/NAS_new/datasets/gta5/zip"
OUTPUT="/media/trunk/NAS_new/datasets/gta5/unzip"
split="train"

help(){
    echo -e "usage:\n ./unzip_.sh split(train, val, test)"
}

if [ $# == 1 ];then
    split=$1
else
    help
    exit 0
fi
echo ${split}
# Move to parent folder
# ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P )"
# cd $ROOT

# Install relative modules
#pip install -r $ROOT/requirements.txt --user

if [ ! -d ${OUTPUT}/${split}/image ];then
    mkdir -p ${OUTPUT}/${split}/image
fi

for i in $(ls ${ROOT}/gta_3d_tracking_${split}_image*.zip);do
    echo $i
    unzip $i -d ${OUTPUT}/${split}/image & # nearly parallel
done
