#!/usr/bin/env bash
file=$1

while read line 
do
  axel -n 20 "$line"
done <$file
