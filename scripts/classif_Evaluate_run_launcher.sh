#!/bin/bash

run_dirs=('outputs/sunscc_run')

job_ids=()
for run_dir in "${run_dirs[@]}"
do
    echo "run_dir: $run_dir"
    job_id=$(sbatch ./scripts/classif_Evaluate_run.sh $run_dir) 
    
    # extract job id
    job_id=$(echo $job_id | grep -o -E '[0-9]+')
    job_ids+=($job_id)
done
echo "job_ids: ${job_ids[@]}"