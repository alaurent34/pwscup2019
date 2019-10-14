#!/usr/bin/env bash
PWD="${PWD}"
DIR="$1"

REGION="../final_phase/OrgData(Final)_011/info_region.csv"

for file in $DIR/*
do
    if [ -f $file ]; then
        IFS='/' read -r -a path <<< "$file"
        IFS='.' read -r -a filename <<< "${path[-1]}"
        mkdir -p "prepro"
        OUT_PATH="$PWD/prepro/prepro_${filename[-2]}/"

        python preprocessing.py -i $file -o $OUT_PATH -r $REGION
    fi
done
