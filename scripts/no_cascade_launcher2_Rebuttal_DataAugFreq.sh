#!/bin/bash


# Parameters
random_move=true
focus_on_group=false

# visual_input=["image","excentricity_map","group_confidence_map"]

use_dtypes=['image','T425-T375-T325_fgbg']
use_npy='rebuttal_all_revised_filtered/all_samples.npy'
# use_npy='rebuttal/all_samples.npy'
use_classes=['A','B','C','SuperGroup','H']

max_epochs=100
training_batches_per_epoch=169 # for all_classes

lr=5e-5
T_max=$((max_epochs*training_batches_per_epoch))
eta_min=1e-8

batch_size=16
num_workers=16

# char_to_balance=('class1' 'class2' 'class3')
char_to_balance=('class1')
# parts_to_train=['encoder','MLP1']

# visual_input=['image','excentricity_map','group_confidence_map']
visual_input=['image','excentricity_map']
# visual_input=['image']

# numeric_input=['angular_excentricity','centroid_Lat']
# numeric_input=['centroid_Lat']
# numeric_input=['angular_excentricity']
numeric_input=[]

j_name='AllRevisedFiltered_Rebuttal_WithHideNoAug' # Rebuttal with the new complete dataset 
# j_name='AllFiltered_Rebuttal_WithHideNoAug' # Rebuttal with the new complete dataset 
# j_name='FINAL_Filtered2_Rebuttal_NoHideNoAug' # Rebuttal with the new complete dataset 
# j_name='Filtered2_Rebuttal_NoHideNoAug' # Rebuttal with the new complete dataset 
# j_name='Filtered2_Rebuttal_NewHideOtherAug' # Rebuttal with the new complete dataset 
# j_name='Rebuttal_NewHideOtherAug' # Rebuttal with the new complete dataset 
# j_name='WithHide_WL-Excentricity_noNumeric' # Rebuttal with the new complete dataset
# j_name='WL-Excentricity_noNumeric_HideOtherAug' # Rebuttal with the new complete dataset + augmentation that removes other groups

data_aug=true
data_aug_type='local_avg'
# data_aug_freq=(0.5 0.75 1.0)
# data_aug_freq=(0.25 0.5 0.75 1.0)
data_aug_freq=(0.0)
# data_aug_freq=(0.25)
# data_aug_freq=(0.5)
# data_aug_freq=(0.75)
# data_aug_freq=(1.0)

# wandb_project='McIntosh_Fast_NoCascade_DataAugFlipRotate'
# cascade=false
cascade=true
wandb_project='Rebuttal'

resnet_type=34

# runids=(21)
# runids=(21 22 23 24 25)
# runids=(26 27 28 29 30)
# runids=(21 22 23 24 25 26 27 28 29 30)
runids=(44)
# runids=(31 32 33 34 35 36 37 38 39 40)

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

            echo "sbatch /home/ucl/elen/nsayez/bio-blueprints/scripts/no_cascade_Rebuttal.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade $resnet_type $runid $data_aug $data_aug_type $da_freq"
            job_id=$(sbatch /home/ucl/elen/nsayez/bio-blueprints/scripts/no_cascade_Rebuttal.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade $resnet_type $runid $data_aug $data_aug_type $da_freq)
        
            # extract job id
            job_id=$(echo $job_id | grep -o -E '[0-9]+')
            job_ids+=($job_id)
        done

    done
done
echo "job_ids: ${job_ids[@]}"

