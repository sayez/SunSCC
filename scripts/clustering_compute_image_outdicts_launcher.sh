#!/bin/bash


look_distance=0.1
kernel_bandwidthLon=( .35)
kernel_bandwidthLat=( 0.04 )
# kernel_bandwidthLon=( 0.1 0.15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75)
# kernel_bandwidthLat=( 0.02 0.04 0.06 .08 .1 .12)
n_iterations=20

num_cpu=32
input_type='confidence_map'

wl_dir=./datasets/classification/2002-2019_2/all
# wl_dir=/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/all
masks_dir=./datasets/classification/2002-2019_2/T425-T375-T325_fgbg
# masks_dir=/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/T425-T375-T325_fgbg
sqlite_path=./datasets/classification/2002-2019_2/drawings_sqlite.sqlite
# sqlite_path=/globalscratch/users/n/s/nsayez/Classification_dataset/drawings_sqlite.sqlite
root_dir=./datasets/classification/2002-2019_2/
# root_dir=/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/
phase1_dir=param_optimization_sunscc
output_dir=param_optimization_P2_sunscc

echo "$PWD"

# ls -l ./scripts/


for i in ${kernel_bandwidthLon[@]}; do
    for j in ${kernel_bandwidthLat[@]}; do
        echo "sbatch --cpus-per-task=$num_cpu ./scripts/clustering_compute_image_outdicts.sh  $input_type $look_distance $i $j  $n_iterations $num_cpu \
        $wl_dir $masks_dir $sqlite_path $root_dir $phase1_dir $output_dir"
        sbatch --cpus-per-task=$num_cpu ./scripts/clustering_compute_image_outdicts.sh  $input_type $look_distance $i $j  $n_iterations $num_cpu \
        $wl_dir $masks_dir $sqlite_path $root_dir $phase1_dir $output_dir
    done
done
