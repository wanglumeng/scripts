#!/usr/bin/env bash

fusionpath=/home/trunk/work/code/metricslidarperception/data/fusion/
metrics_script_path=/home/trunk/work/code/metricslidarperception/
metrics_script_root=${metrics_script_path}/metrics
metrics_input_data_root=/home/trunk/work/code/metricslidarperception/
metrics_output_data_root=/home/trunk/work/code/metricslidarperception/result/
sed -i -r 's%sensor_type: .*%sensor_type: '"\"fusion\""'%' $metrics_script_root/config/object_eval.yaml
sed -i -r 's%gt_path: .*%gt_path: '"\"${metrics_input_data_root}/data/gt_at128_v4\""'%' $metrics_script_root/config/object_eval.yaml
sed -i -r 's%pred_path: .*%pred_path: '"\"${fusionpath}\""'%' $metrics_script_root/config/object_eval.yaml
sed -i -r 's%data_type: .*%data_type: '"\"tracked_objs\""'%' $metrics_script_root/config/object_eval.yaml
sed -i -r 's%metrics_result_path: .*%metrics_result_path: '"\"${metrics_output_data_root}\""'%' $metrics_script_root/config/object_eval.yaml

python /home/trunk/work/code/metricslidarperception/metrics/eval/track_evaluate.py
