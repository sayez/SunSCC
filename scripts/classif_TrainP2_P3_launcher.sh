#!/bin/bash

run_dirs=('/home/ucl/elen/nsayez/sunscc/outputs/2023-08-11/18-36-37/run_0.0_class1_100epochs_run0')

#save pwd in a variable
p=$(pwd)
# set an environment variable for dataset path
export SUNSCCDATASET_PATH=$p
echo "SUNSCCDATASET_PATH: $SUNSCCDATASET_PATH"

for run_dir in "${run_dirs[@]}"
do
    echo "run_dir: $run_dir"
    sbatch ./scripts/classif_TrainP2_P3.sh $run_dir
done