#!/usr/bin/env bash
path=$(dirname $(readlink -f "$0"))
for dir in $path/*; do
	if [ -d $dir/.git ]; then
		pushd $dir
		git pull
		popd
	else
		for dir1 in $dir/src/*; do
			if [ -d $dir1/.git ]; then
				pushd $dir1
				git pull
				popd
			fi
		done
	fi
done
