#!/usr/bin/env bash
PWD="${PWD}"
REF="$1"
ANO="$2"

REGION="../preliminary_phase/OrgData(Anony-Pre)_011/info_region.csv"

for file in $REF/*
do
    if [ -f $file ]; then
        IFS='/' read -r -a path <<< "$file"
        IFS='.' read -r -a filename <<< "${path[-1]}"
        OUT_PATH="$PWD/prepro_${filename[-2]}/"

        python preprocessing.py -i $file -o $OUT_PATH -r $REGION
    fi
done

for file in $ANO/*
do
    if [ -f $file ]; then
        IFS='/' read -r -a path <<< "$file"
        IFS='.' read -r -a filename <<< "${path[-1]}"
        OUT_PATH="$PWD/prepro_${filename[-2]}/"

        python preprocessing.py -i $file -o $OUT_PATH -r $REGION
    fi
done
