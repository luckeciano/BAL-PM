#!/bin/bash

for i in $(seq 0 $2);
do
    sbatch -J $1_$i run_rwft_gpt2xl.sh $1 $i
    sleep 60
done
