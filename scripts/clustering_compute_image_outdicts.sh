#!/bin/bash

############## Parameters
#######S#B#A#T#C#H --cpus-per-task=40


# ATTENTION: the SBATCH --jobname is set in the calling script
# in the job_name variable (argindex 5)

#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=1800

# S%B%A%TCH --exclude=mb-cas001,mb-cas101,mb-mil102


# command
export SUBMITIT_EXECUTOR=slurm

input_type=$1
look_distance=$2
kernel_bandwidthLon=$3
kernel_bandwidthLat=$4
n_iterations=$5

num_cpu=$6

wl_dir=$7
masks_dir=$8
sqlite_path=$9
root_dir=${10}
phase1_dir=${11}
output_dir=${12}

#print the hostname
echo "hostname: $(hostname)"

echo "num_cpu: $num_cpu input_type: $input_type  look_distance: $look_distance kernel_bandwidthLon: $kernel_bandwidthLon kernel_bandwidthLat: $kernel_bandwidthLat n_iterations: $n_iterations"

python ./utilities/clustering_compute_image_outdict.py \
        --wl_dir $wl_dir --masks_dir $masks_dir --sqlite_db_path $sqlite_path \
        --root_dir $root_dir --phase1_dir $phase1_dir --output_dir $output_dir \
        --input_type $input_type \
         --look_distance $look_distance --kernel_bandwidthLon $kernel_bandwidthLon --kernel_bandwidthLat $kernel_bandwidthLat \
         --n_iterations $n_iterations --num_cpu $num_cpu

