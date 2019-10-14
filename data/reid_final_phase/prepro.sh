#!/usr/bin/env bash
PWD="${PWD}"
DIR="$1"

for file in $DIR/*
do
    if [ -f $file ]; then
        IFS='/' read -r -a path <<< "$file"
        IFS='.' read -r -a filename <<< "${path[-1]}"
        mkdir -p "prepro"
        OUT_PATH="$PWD/prepro/prepro_${filename[-2]}/"

        python preprocessing.py -i $file -o $OUT_PATH -v
    fi
done
