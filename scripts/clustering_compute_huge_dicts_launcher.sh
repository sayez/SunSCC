#!/bin/bash

look_distance=0.1
kernel_bandwidthLon=( 0.1 0.15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75)
kernel_bandwidthLat=( .02 .12)
kernel_bandwidthLat=( 0.02 0.04 0.06 .08 .1 .12)
n_iterations=20

num_cpu=32
input_type='confidence_map'


for i in ${kernel_bandwidthLon[@]}; do
    for j in ${kernel_bandwidthLat[@]}; do
        sbatch --cpus-per-task=$num_cpu ./sunscc/scripts/compute_huge_dicts.sh  $input_type $look_distance $i $j  $n_iterations $num_cpu
    done
done
