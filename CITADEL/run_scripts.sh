#! /bin/bash

## Script to run CITADEL with active learning on the API-Graph dataset

# DATASET="apigraph"
# DATA_DIR="/home/mhaque3/myDir/data/gen_apigraph_drebin/"
# # UC='boundary'
# UC='priority'
# BUDGET=400
# AUG="random_bit_flip_bernoulli"
# SEED=220
# LR=0.03

# EPOCHS=200
# RETRAIN_EPOCHS=70

# BATCH_SIZE=512
# AL_BATCH_SIZE=512


# AL_START_YEAR=2013
# AL_START_MONTH=7
# AL_END_YEAR=2018
# AL_END_MONTH=12

# strategy="CITADEL"
# LABEL_RATIO=0.4
# LAMBDA_SUPCON=0.5
# TS=$(date "+%m.%d-%H.%M.%S")

# SAVE_PATH="CITADEL_results"
# mkdir -p $SAVE_PATH
# CUDA_VISIBLE_DEVICES=1 nohup python -u citadel_fixmatch_al.py \
#         --dataset $DATASET --data_dir $DATA_DIR \
#         --lr $LR \
#         --epochs $EPOCHS --retrain_epochs $RETRAIN_EPOCHS \
#         --labeled_ratio $LABEL_RATIO \
#         --lambda_supcon $LAMBDA_SUPCON \
#         --batch_size $BATCH_SIZE \
#         --al_batch_size $AL_BATCH_SIZE \
#         --al_start_year $AL_START_YEAR --al_start_month $AL_START_MONTH \
#         --al_end_year $AL_END_YEAR --al_end_month $AL_END_MONTH \
#         --al \
#         --supcon \
#         --unc_samp $UC \
#         --budget $BUDGET \
#         --aug $AUG \
#         --seed $SEED \
#         --strategy $strategy \
#         --save_path $SAVE_PATH > $SAVE_PATH/citadel_w_al_${UC}_${AUG}_${BUDGET}_seed_${SEED}_${TS}.log 2>&1 &
    


## Script to run CITADEL with active learning on the Chen-AndroZoo dataset
DATASET="chen-androzoo"
DATA_DIR="/home/mhaque3/myDir/data/gen_androzoo_drebin/"
# UC='boundary'
UC='priority'
BUDGET=400
AUG="random_bit_flip_bernoulli"
SEED=220
LR=0.003

EPOCHS=2
RETRAIN_EPOCHS=7

BATCH_SIZE=128
AL_BATCH_SIZE=128


AL_START_YEAR=2020
AL_START_MONTH=7
AL_END_YEAR=2021
AL_END_MONTH=12

strategy="CITADEL"
LABEL_RATIO=0.4
LAMBDA_SUPCON=0.5
TS=$(date "+%m.%d-%H.%M.%S")

SAVE_PATH="CITADEL_results"
mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=2 nohup python -u citadel_fixmatch_al.py \
        --dataset $DATASET --data_dir $DATA_DIR \
        --lr $LR \
        --epochs $EPOCHS --retrain_epochs $RETRAIN_EPOCHS \
        --labeled_ratio $LABEL_RATIO \
        --lambda_supcon $LAMBDA_SUPCON \
        --batch_size $BATCH_SIZE \
        --al_batch_size $AL_BATCH_SIZE \
        --al_start_year $AL_START_YEAR --al_start_month $AL_START_MONTH \
        --al_end_year $AL_END_YEAR --al_end_month $AL_END_MONTH \
        --al \
        --supcon \
        --unc_samp $UC \
        --budget $BUDGET \
        --aug $AUG \
        --seed $SEED \
        --strategy $strategy \
        --save_path $SAVE_PATH > $SAVE_PATH/citadel_w_al_${UC}_${AUG}_${BUDGET}_seed_${SEED}_${TS}.log 2>&1 &
    


## Script to run FixMatch with active learning on the LAMDA dataset
DATASET="lamda"
DATA_DIR="/home/shared-datasets/Feature_extraction/Our_experiments/LAMDA_dataset/Baseline_npz/"
# UC='boundary'
UC='priority'
BUDGET=400
AUG="random_bit_flip_bernoulli"
SEED=220
LR=0.03

EPOCHS=2
RETRAIN_EPOCHS=7

BATCH_SIZE=128
AL_BATCH_SIZE=128


AL_START_YEAR=2014
AL_START_MONTH=7
AL_END_YEAR=2025
AL_END_MONTH=1

strategy="CITADEL"
LABEL_RATIO=0.4
LAMBDA_SUPCON=0.5
TS=$(date "+%m.%d-%H.%M.%S")

SAVE_PATH="CITADEL_results"
mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=2 nohup python -u citadel_fixmatch_al.py \
        --dataset $DATASET --data_dir $DATA_DIR \
        --lr $LR \
        --epochs $EPOCHS --retrain_epochs $RETRAIN_EPOCHS \
        --labeled_ratio $LABEL_RATIO \
        --lambda_supcon $LAMBDA_SUPCON \
        --batch_size $BATCH_SIZE \
        --al_batch_size $AL_BATCH_SIZE \
        --al_start_year $AL_START_YEAR --al_start_month $AL_START_MONTH \
        --al_end_year $AL_END_YEAR --al_end_month $AL_END_MONTH \
        --al \
        --supcon \
        --unc_samp $UC \
        --budget $BUDGET \
        --aug $AUG \
        --seed $SEED \
        --strategy $strategy \
        --save_path $SAVE_PATH > $SAVE_PATH/citadel_w_al_${UC}_${AUG}_${BUDGET}_seed_${SEED}_${TS}.log 2>&1 &
    
