#! /bin/bash

## Script to run CITADEL without active learning on the API-Graph dataset


LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
AUGS=("random_bit_flip_bernoulli" "random_bit_flip" "random_bit_flip_and_mask" "random_feature_mask")

DATASET="apigraph"   # change to "chen-androzoo" or "lamda" as needed
DATA_DIR="/home/mhaque3/myDir/data/gen_apigraph_drebin/" # change to appropriate data directory
BATCH_SIZE=512
LR=0.03

mkdir -p CITADEL_table2

for AUG in "${AUGS[@]}"
do
    for RATIO in "${LABEL_RATIOS[@]}"
    do
        TS=$(date "+%m.%d-%H.%M.%S")
        LOGFILE="CITADEL_table2/citadel_no_al_${DATASET}_${AUG}_ratio_${RATIO}_${TS}.log"
        echo "Running: Dataset=$DATASET | Aug=$AUG | Ratio=$RATIO"

        CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_main.py \
            --dataset $DATASET \
            --data_dir $DATA_DIR \
            --lr $LR \
            --epochs 200 \
            --batch_size $BATCH_SIZE \
            --labeled_ratio $RATIO \
            --lambda_supcon 0.5 \
            --supcon \
            --aug $AUG \
            --seed 220 \
            --strategy CITADEL \
            --save_path CITADEL_table2 > $LOGFILE 2>&1 &
    done
done


## Script to run CITADEL without active learning on the Chen-AndroZoo dataset
LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
AUGS=("random_bit_flip_bernoulli" "random_bit_flip" "random_bit_flip_and_mask" "random_feature_mask")

DATASET="chen-androzoo"
DATA_DIR="/home/mhaque3/myDir/data/gen_androzoo_drebin/"
BATCH_SIZE=512
LR=0.03

mkdir -p CITADEL_table2

for AUG in "${AUGS[@]}"
do
    for RATIO in "${LABEL_RATIOS[@]}"
    do
        TS=$(date "+%m.%d-%H.%M.%S")
        LOGFILE="CITADEL_table2/citadel_no_al_${DATASET}_${AUG}_ratio_${RATIO}_${TS}.log"
        echo "Running: Dataset=$DATASET | Aug=$AUG | Ratio=$RATIO"

        CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_main.py \
            --dataset $DATASET \
            --data_dir $DATA_DIR \
            --lr $LR \
            --epochs 200 \
            --batch_size $BATCH_SIZE \
            --labeled_ratio $RATIO \
            --lambda_supcon 0.5 \
            --supcon \
            --aug $AUG \
            --seed 220 \
            --strategy CITADEL \
            --save_path CITADEL_table2 > $LOGFILE 2>&1 &
    done
done


## Script to run CITADEL without active learning on the LAMDA dataset
LABEL_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
AUGS=("random_bit_flip_bernoulli" "random_bit_flip" "random_bit_flip_and_mask" "random_feature_mask")

DATASET="lamda"
DATA_DIR="./LAMDA_dataset/NPZ_Version/npz_Baseline"
BATCH_SIZE=512
LR=0.03

mkdir -p CITADEL_table2

for AUG in "${AUGS[@]}"
do
    for RATIO in "${LABEL_RATIOS[@]}"
    do
        TS=$(date "+%m.%d-%H.%M.%S")
        LOGFILE="CITADEL_table2/citadel_no_al_${DATASET}_${AUG}_ratio_${RATIO}_${TS}.log"
        echo "Running: Dataset=$DATASET | Aug=$AUG | Ratio=$RATIO"

        CUDA_VISIBLE_DEVICES=1 nohup python -u CITADEL/citadel_main.py \
            --dataset $DATASET \
            --data_dir $DATA_DIR \
            --lr $LR \
            --epochs 200 \
            --batch_size $BATCH_SIZE \
            --labeled_ratio $RATIO \
            --lambda_supcon 0.5 \
            --supcon \
            --aug $AUG \
            --seed 220 \
            --strategy CITADEL \
            --save_path CITADEL_table2 > $LOGFILE 2>&1 &
    done
done