#!/usr/bin/env bash
# PWD="${PWD}"
DIR="$1"

for file in $DIR/*
do
    if [ -d $file ]; then
        IFS='/' read -r -a path <<< "$file"
        IFS='.' read -r -a filename <<< "${path[-1]}"
        IFS='_' read -r -a ftype <<< "${filename[0]}"

        ftype="${ftype[1]}"
        if [ $ftype == "reftraces" ]; then
            continue
        fi

        REF_PATH="$DIR/prepro_reftraces_${ftype[-3]}_${ftype[-2]}_${ftype[-1]}"
        ANO_PATH="$file/"

        python attack.py -o $ANO_PATH -r $REF_PATH -j -s 2
    fi
done
