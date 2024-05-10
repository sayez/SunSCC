#!/bin/bash

#save pwd in a variable
p=$(pwd)
# set an environment variable for dataset path
export SUNSCCDATASET_PATH=$p
echo "SUNSCCDATASET_PATH: $SUNSCCDATASET_PATH"

run_dirs=("$SUNSCCDATASET_PATH/outputs/2024-05-10/11-47-01/run_0.0_class1_100epochs_run0")

p2_p3_max_epochs=100 # number of epochs each phase must train

for run_dir in "${run_dirs[@]}"
do
    echo "run_dir: $run_dir"
    echo "sbatch ./scripts/classif_TrainP2_P3.sh $run_dir $p2_p3_max_epochs"
    sbatch ./scripts/classif_TrainP2_P3.sh $run_dir $p2_p3_max_epochs
done