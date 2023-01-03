#!/usr/bin/env bash
##### help
# ./play_scenebag.sh # 播放所有bag
# ./play_scenebag.sh <index> # 播放第几个bag，0对应1_1
# 流程：（播放所有bag时）
# 1. build: 修改calibration.cpp标定参数，重新编译
# 2. output: 修改源码里的输出路径为${output_dir}，同时打开输出开关
# 3. launch: play其他topic和lidar目标的bag，roslaunch启动融合节点
# 4. save_json: 杀掉融合节点，将本次输出的txt/json文件移到metrics需要的路径（${output_dir}/*.bag/）
# (播放单个bag时)
# launch： play其他topic和lidar目标的bag， 杀掉融合节点（曾经出现过卡死）

index=0
scene=""
if [ $# == 1 ]; then
  if [[ $1 =~ "_" ]]; then # 传入的参数有下划线，直接作为scene?_?的文件名
    scene=$1
  else # 传入的参数没有下划线，作为序号
    index=$1
  fi
  echo $1
fi
# dir="/media/trunk/sata/bag/hota_0613/output"
# obj_dir="/media/trunk/sata/bag/hota_0613/results"
dir="/media/trunk/sata/bag/hota_0613/bag_data_without_obj" # 除了lidar目标其他topic的bag路径
obj_dir="/media/trunk/sata/bag/hota_0613/perception_objs"  # lidar目标bag路径
suffix="*.bag"
output_dir="/home/trunk/work/code/metricslidarperception/data/fusion" # 输出路径，metrics仓库里的位置
sensorfusion_dir="/home/trunk/work/code/sensorfusion_ws/"             # 源码路径，会修改calibration.cpp里的标定参数，再catkin build编译
bags=(${dir}/scene1_1${suffix}
  ${dir}/scene1_3${suffix}
  ${dir}/scene1_4${suffix}
  ${dir}/scene2_1${suffix}
  ${dir}/scene2_2${suffix}
  ${dir}/scene2_3${suffix}
  ${dir}/scene3_1${suffix}
  ${dir}/scene3_2${suffix}
  ${dir}/scene3_3${suffix}
  ${dir}/scene4_1${suffix}
  ${dir}/scene4_2${suffix}
  ${dir}/scene4_3${suffix}
  ${dir}/scene5_1${suffix}
  ${dir}/scene5_3${suffix}
  ${dir}/scene6_1${suffix}
  ${dir}/scene6_3${suffix}
  ${dir}/scene6_5${suffix}
  ${dir}/scene6_6${suffix}
  ${dir}/scene6_8${suffix}
  ${dir}/scene7_1${suffix}
  ${dir}/scene7_2${suffix}
  ${dir}/scene7_3${suffix}
  ${dir}/scene8_1${suffix}
  ${dir}/scene8_2${suffix}
  ${dir}/scene8_3${suffix}
)
objs=(${obj_dir}/scene1_1${suffix}
  ${obj_dir}/scene1_3${suffix}
  ${obj_dir}/scene1_4${suffix}
  ${obj_dir}/scene2_1${suffix}
  ${obj_dir}/scene2_2${suffix}
  ${obj_dir}/scene2_3${suffix}
  ${obj_dir}/scene3_1${suffix}
  ${obj_dir}/scene3_2${suffix}
  ${obj_dir}/scene3_3${suffix}
  ${obj_dir}/scene4_1${suffix}
  ${obj_dir}/scene4_2${suffix}
  ${obj_dir}/scene4_3${suffix}
  ${obj_dir}/scene5_1${suffix}
  ${obj_dir}/scene5_3${suffix}
  ${obj_dir}/scene6_1${suffix}
  ${obj_dir}/scene6_3${suffix}
  ${obj_dir}/scene6_5${suffix}
  ${obj_dir}/scene6_6${suffix}
  ${obj_dir}/scene6_8${suffix}
  ${obj_dir}/scene7_1${suffix}
  ${obj_dir}/scene7_2${suffix}
  ${obj_dir}/scene7_3${suffix}
  ${obj_dir}/scene8_1${suffix}
  ${obj_dir}/scene8_2${suffix}
  ${obj_dir}/scene8_3${suffix}
)

function calibration() {
  # input param: calibration.cpp file path
  file=$1
  sed -i -r 's%define TRANSLATION_X.*%define TRANSLATION_X 5.46%' ${file}
  sed -i -r 's%define TRANSLATION_Y.*%define TRANSLATION_Y 0.0%' ${file}
  sed -i -r 's%define TRANSLATION_Z.*%define TRANSLATION_Z 0.65%' ${file}
  sed -i -r 's%define ROTATION_ROLL.*%define ROTATION_ROLL 0.0%' ${file}
  sed -i -r 's%define ROTATION_PITCH.*%define ROTATION_PITCH 0.0%' ${file}
  sed -i -r 's%define ROTATION_YAW.*%define ROTATION_YAW 0.48%' ${file}
}

function build() {
  calibration /home/trunk/work/code/sensorfusion_ws/src/sensorfusion/src/src/sensors/calibration/calibration.cpp
  pushd /home/trunk/work/code/sensorfusion_ws/
  catkin clean -y
  catkin build
  popd
}

function launch() {
  # input index1 index2
  if [ $# == 2 ]; then
    roslaunch sensor_fusion_core sensor_fusion_offline.launch &
    sleep 12
    rosbag play --clock ${bags[${1}]} ${objs[${2}]} /sensor_fusion_core/FusionObject:=/lll
    # /dev/video0/compressed:=/v0 /dev/video1/compressed:=/v1 \
    # /dev/video2/compressed:=/v2 /dev/video3/compressed:=/v3 \
    # /dev/video4/compressed:=/v4 /dev/video5/compressed:=/v5 \
    # /dev/video6/compressed:=/v6 /dev/video7/compressed:=/v7
  elif [ $# == 1 ]; then
    rosbag play --clock ${bags[${1}]} ${objs[${1}]} /sensor_fusion_core/FusionObject:=/lll
    # /dev/video0/compressed:=/v0 /dev/video1/compressed:=/v1 \
    # /dev/video2/compressed:=/v2 /dev/video3/compressed:=/v3 \
    # /dev/video4/compressed:=/v4 /dev/video5/compressed:=/v5 \
    # /dev/video6/compressed:=/v6 /dev/video7/compressed:=/v7
  fi
}

function launch_str() {
  # input scene
  rosbag play --clock ${dir}/scene${1}${suffix} ${obj_dir}/scene${1}${suffix} /sensor_fusion_core/FusionObject:=/lll
  # /dev/video0/compressed:=/v0 /dev/video1/compressed:=/v1 \
  # /dev/video2/compressed:=/v2 /dev/video3/compressed:=/v3 \
  # /dev/video4/compressed:=/v4 /dev/video5/compressed:=/v5 \
  # /dev/video6/compressed:=/v6 /dev/video7/compressed:=/v7
}

function save_txt() {
  # input bag_name
  file=${1}.txt
  rosnode kill /sensor_fusion_core

  sleep 1
  mkdir -p ${output_dir}
  mv ${output_dir}/sensor_fusion_output.txt ${output_dir}/${file}
}

function save_json() {
  # input bag_name
  dir=${1}.bag
  echo ${dir}
  rosnode kill /sensor_fusion_core

  sleep 1
  rm -r ${output_dir}/${dir}/
  mkdir -p ${output_dir}/${dir}/
  mv ${output_dir}/*.json ${output_dir}/${dir}/
}

function output() {
  sed -i -r 's%logOn.*%logOn = true%' ${sensorfusion_dir}/src/sensorfusion/params/track_config.toml
  sed -i -r 's%logPath.*%logPath = '\"${output_dir}\"'%' ${sensorfusion_dir}/src/sensorfusion/params/track_config.toml
  source ${sensorfusion_dir}/devel/setup.bash
  rm -r ${output_dir}/*.json # rm temp output dir
}

if [ $# == 1 ]; then # bag(index)
  echo ${scene}
  if [ ${scene} ]; then
    launch_str ${scene} # str
  else
    launch ${index} # index(int)
  fi
  rosnode kill /sensor_fusion_core

  echo ${bags[${index}]}
  file=${bags[${index}]##*\/}
  # save_json ${file::-4}
else # bag(all)
  ## rebuild
  # build
  # output
  i=0
  while [[ i -lt ${#bags[@]} ]]; do
    clear
    launch i i

    echo ${bags[${i}]}
    file=${bags[${i}]##*\/}

    # save_txt ${file::-4}
    save_json ${file::-4}
    let i++
  done
fi
