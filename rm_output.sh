#!/usr/bin/env bash

# Specify the root path where you want to search for 'output' directories
root_path="/home/trunk/work/code/target_fusion_lib"

# Use the 'find' command to locate all 'output' directories
find "$root_path" -type d -name "output" | while read -r output_dir; do
  # Check if the directory exists before removing it
  if [ -d "$output_dir" ]; then
    echo "Removing directory: $output_dir"
    rm -r "$output_dir"
  fi
done
