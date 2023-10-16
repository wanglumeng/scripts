#!/usr/bin/env bash
roscore&
sleep 1
ls ~/work/code/tad_soc_release/entry/config/*.yaml | xargs -I @ rosparam load @ 
rosparam set /function/log/level 4
