#!/bin/bash

############## Parameters
#######S#B#A#T#C#H --cpus-per-task=40


# ATTENTION: the SBATCH --jobname is set in the calling script
# in the job_name variable (argindex 5)

#SBATCH --mem-per-cpu=5GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=1800

# command
export SUBMITIT_EXECUTOR=slurm

input_type=$1
look_distance=$2
kernel_bandwidthLon=$3
kernel_bandwidthLat=$4
n_iterations=$5

num_cpu=$6

#print the hostname
echo "hostname: $(hostname)"

echo "num_cpu: $num_cpu input_type: $input_type  look_distance: $look_distance kernel_bandwidthLon: $kernel_bandwidthLon kernel_bandwidthLat: $kernel_bandwidthLat n_iterations: $n_iterations"

python ./scripts/compute_huge_dict.py \
        --input_type $input_type \
         --look_distance $look_distance --kernel_bandwidthLon $kernel_bandwidthLon --kernel_bandwidthLat $kernel_bandwidthLat \
         --n_iterations $n_iterations --num_cpu $num_cpu

