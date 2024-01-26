#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --constraint=TeslaA100_80|TeslaA100


# ATTENTION: the SBATCH --jobname is set in the calling script
# in the job_name variable (argindex 5)


#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=06:00:00

run_dir=$1

echo "run_dir: $run_dir"

#save pwd in a variable
p=$(pwd)
# set an environment variable for dataset path
export SUNSCCDATASET_PATH=$p
echo "SUNSCCDATASET_PATH: $SUNSCCDATASET_PATH"

python ./scripts/classif_TrainP2_P3.py --run_dir $run_dir
