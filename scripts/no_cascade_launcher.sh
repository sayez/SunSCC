#!/bin/bash


# Parameters
random_move=true
focus_on_group=false

# visual_input=["image","excentricity_map","group_confidence_map"]

use_dtypes=['image','T425-T375-T325_fgbg']
use_npy='test/all_samples.npy'
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

# j_name='WL-Excentricity-conf'
# j_name='WL-Excentricity'
# j_name='WLonly'
# j_name='WL-Excentricity-conf_noNumeric'
j_name='WL-Excentricity_noNumeric'
# j_name='WLonly_noNumeric'

# wandb_project='McIntosh_Fast_NoCascade_DataAugFlipRotate'
# cascade=false
cascade=true
# wandb_project='McIntosh_Fast_Cascade_DataAugFlipRotate'
wandb_project='test_mil102'

for c in "${char_to_balance[@]}"
do
    echo "c: $c"
    job_name="${j_name}_${c}_${max_epochs}epochs"
    echo "job_name: $job_name"
    
    if [ "$c" = "class1" ]; then 
        parts_to_train=['encoder','MLP1']
    elif [ "$c" = "class2" ]; then
        parts_to_train=['encoder','MLP2']
    elif [ "$c" = "class3" ]; then
        parts_to_train=['encoder','MLP3']
    fi

    echo "parts_to_train: $parts_to_train"

    echo "sbatch ./sunscc/scripts/no_cascade.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade"
    sbatch ./sunscc/scripts/no_cascade.sh $random_move $focus_on_group "$use_dtypes" $use_npy "$use_classes" $max_epochs "$parts_to_train" $lr $T_max $eta_min $batch_size $num_workers $c $job_name $wandb_project "$visual_input" "$numeric_input" $cascade
done
