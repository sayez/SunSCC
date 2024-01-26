#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --constraint=TeslaA100_80|TeslaA100

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=06:00:00


run_dir=$1
use_npy=$2
outpath=$3

echo "run_dir: $run_dir"

srun python ./scripts/classif_Evaluate_run.py --run_dir $run_dir --outpath $outpath --use_npy $use_npy