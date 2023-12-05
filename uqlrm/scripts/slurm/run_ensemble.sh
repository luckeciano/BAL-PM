#!/bin/bash

for i in $(seq 0 $2):
do
    sbatch -J $1_$i run_rwft.sh $1 $i
done