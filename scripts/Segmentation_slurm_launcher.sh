#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=32

##### Attention: SBATCH --gres=gpu:TeslaA100_80:1
#SBATCH --gres=gpu:TeslaA100:1

# ATTENTION: the SBATCH --jobname is set in the calling script
# in the job_name variable (argindex 5)

#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=1800

# command
export SUBMITIT_EXECUTOR=slurm

# Script arguments
# $1: num_workers
# $2: use_dtypes
# $3: max_epochs
# $4: project_name
# $5: job_name (Wandb run name)


num_workers=$1
use_dtypes=$2
max_epochs=$3
project_name=$4
job_name=$5
scheduler_type=$6
scheduler_interval=$7
scheduler_step_size=$8
run_id=$9
classes=${10}


dt=$(date '+%Y-%m-%d/%H-%M-%S');
hydra_out_dir="./outputs/${dt}_${job_name}"

# waittime=$((RANDOM % 20))
# echo "sleeping for $waittime seconds"
# sleep $waittime

echo "num_workers: $num_workers, classes:$classes, use_dtypes: $use_dtypes, max_epochs: $max_epochs, project_name: $project_name, job_name: $job_name, scheduler_type: $scheduler_type, scheduler_interval: $scheduler_interval, scheduler_step_size: $scheduler_step_size"
echo "run_id: $run_id "

#save pwd in a variable
p=$(pwd)

# set an environment variable for dataset path
export SUNSCCDATASET_PATH=$p
echo "SUNSCCDATASET_PATH: $SUNSCCDATASET_PATH"

if [ "$scheduler_type" = "MultistepLR" ]; then
    echo "MultistepLR"
    # Utiliser exp=2023_2013-15_CombineLoss pour utiliser le scheduler MultistepLR
    python -m sunscc.train exp=Segmentation_MultistepLR_CombineLoss gpus=1 \
                hydra.run.dir=$hydra_out_dir \
                seed=$run_id \
                use_dtypes=$use_dtypes \
                trainer.max_epochs=$max_epochs \
                model.classes=$classes \
                dataset.num_workers=$num_workers \
                logger.0.project=$project_name +logger.0.name=$job_name
elif [ "$scheduler_type" = "StepLR" ]; then
    echo "StepLR"
    # Utiliser exp=2013-15_CombineLoss pour utiliser le scheduler StepLR
    # On peut changer le scheduler_interval et le scheduler_step_size
    # (si scheduler_step_size > max_epochs, alors on utilise le scheduler StepLR)
    python -m sunscc.train exp=Segmentation_StepLR_CombineLoss gpus=1 \
                hydra.run.dir=$hydra_out_dir \
                seed=$run_id \
                use_dtypes=$use_dtypes \
                trainer.max_epochs=$max_epochs \
                model.classes=$classes \
                dataset.num_workers=$num_workers \
                logger.0.project=$project_name +logger.0.name=$job_name \
                module.scheduler_interval=$scheduler_interval \
                module.scheduler.step_size=$scheduler_step_size
else
    echo "No LR scheduler"
    # No LR scheduler = StepLR scheduler (step_size> max_epochs)
    python -m sunscc.train exp=Segmentation_StepLR_CombineLoss gpus=1 \
                hydra.run.dir=$hydra_out_dir \
                seed=$run_id \
                use_dtypes=$use_dtypes \
                trainer.max_epochs=$max_epochs \
                model.classes=$classes \
                dataset.num_workers=$num_workers \
                logger.0.project=$project_name +logger.0.name=$job_name 
fi

output_file="$SLURM_JOB_NAME-$SLURM_JOB_ID.out"
echo $output_file

run_dir=$(find outputs/ -name  "$(grep 'wandb: Run data is saved locally in ./wandb/' $output_file | sed -r 's/^([^ ]+ ){7}.\/wandb\///')" | sed 's/wandb\/.*//' )
echo $run_dir

mv $output_file $run_dir

