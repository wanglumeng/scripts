#!/usr/bin/env bash

function run_metrics() {
    path=${1}
    scene_list=$(ls $1) # $1 common bag path, scene:common/rain
    metrics_script_root=$2
    metrics_gt_path=$3
    metrics_output_data_root=$4
    metrics_script_path=$5

    for scene in ${scene_list}; do
        mkdir -p ${metrics_output_data_root}/${scene}
        # 删除perf.json和0.json
        rm ${fusionpath}/${scene}/*/0.json
        rm ${fusionpath}/${scene}/*/perf.json
        sed -i -r 's%sensor_type: .*%sensor_type: '"\"fusion\""'%' ${metrics_script_root}/config/object_eval.yaml
        sed -i -r 's%gt_path: .*%gt_path: '"\"${metrics_gt_path}/${scene}\""'%' ${metrics_script_root}/config/object_eval.yaml
        sed -i -r 's%pred_path: .*%pred_path: '"\"${fusionpath}/${scene}\""'%' ${metrics_script_root}/config/object_eval.yaml
        sed -i -r 's%data_type: .*%data_type: '"\"tracked_objs\""'%' ${metrics_script_root}/config/object_eval.yaml
        sed -i -r 's%metrics_result_path: .*%metrics_result_path: '"\"${metrics_output_data_root}/${scene}\""'%' ${metrics_script_root}/config/object_eval.yaml
        python3 "${metrics_script_path}"/metrics/eval/track_evaluate.py
    done
}

path=/media/trunk/sata/bag/hota_0613/scenes/common_bag/
fusionpath=/home/trunk/work/code/metricslidarperception/data/fusion/
metrics_script_path=/home/trunk/work/code/metricslidarperception/
metrics_script_root=${metrics_script_path}/metrics
metrics_gt_path=/media/trunk/sata/bag/hota_0613/scenes/gt/
metrics_output_data_root=/home/trunk/work/code/metricslidarperception/result/

run_metrics ${path} ${metrics_script_root} ${metrics_gt_path} ${metrics_output_data_root} ${metrics_script_path}
