#!/bin/bash


# Parameters
random_move=true
focus_on_group=false


use_dtypes=['image','T425-T375-T325_fgbg']
use_npy='rebuttal_all_revised_filtered/all_samples.npy' # Do not set the _train.npy, _val.npy or test.npy,
                                                        # it is added automatically in the dataloader.

use_classes=['A','B','C','SuperGroup','H']

max_epochs=100
training_batches_per_epoch=169 # for all_classes

lr=5e-5
T_max=$((max_epochs*training_batches_per_epoch))
eta_min=1e-8

batch_size=16
num_workers=16

char_to_balance=('class1')  # We train only the Z component of the classifier

visual_input=['image','excentricity_map']

numeric_input=[]

j_name='run' # Prefix of the job name (used for wandb run name) and
             # the name of the folder where the model is saved.
             
# Parameters for data augmentation
data_aug=true
data_aug_type='local_avg'
data_aug_freq=(0.0)

cascade=true
wandb_project='sunscc'

resnet_type=34

runids=(0) # id numbers of the runs to perform,
           # set several values to get an ensemble of classifiers

job_ids=()

for da_freq in "${data_aug_freq[@]}"
do
    echo "da_freq: $da_freq"

    for runid in "${runids[@]}"
    do
        for c in "${char_to_balance[@]}"
        do
            echo "c: $c"
            job_name="${j_name}_${da_freq}_${c}_${max_epochs}epochs_run${runid}"
            echo "job_name: $job_name"
            
            if [ "$c" = "class1" ]; then 
                parts_to_train=['encoder','MLP1']
            elif [ "$c" = "class2" ]; then
                parts_to_train=['encoder','MLP2']
            elif [ "$c" = "class3" ]; then
                parts_to_train=['encoder','MLP3']
            fi

            echo "parts_to_train: $parts_to_train"

            echo "./scripts/classif_TrainP1.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade $resnet_type $runid $data_aug $data_aug_type $da_freq"
            job_id=$(sbatch ./scripts/classif_TrainP1.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade $resnet_type $runid $data_aug $data_aug_type $da_freq)
        
            # extract job id
            job_id=$(echo $job_id | grep -o -E '[0-9]+')
            job_ids+=($job_id)
        done

    done
done
echo "job_ids: ${job_ids[@]}"

