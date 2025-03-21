#! /bin/bash

SEQ=088
LR=0.0001
OPT=adam
SCH=step
DECAY=0.90
EPOCHS=100
RESULT_EPOCHS=30
WLR=0.00015
WE=0.0001
# DATA=/home/ihossain/ISMAIL/Datasets/data/gen_androzoo_drebin
DATA=/home/ihossain/ISMAIL/Datasets/data/gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12

VALID_START=2013-01
VALID_END=2013-01

TEST_START=2013-07
TEST_END=2013-07

SM_FN='cosine'
NUM_SAMP=1000
LAMBDA=0.1
K=1

RESULT_DIR=/home/ihossain/ISMAIL/SSL-malware/results_ours
AL_OPT=adam
PRE=False
GPU=cuda:2

CNT=1000

modeldim="512-384-256-128"
S='half'
B=1024
LOSS='ssl-loss'
TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u main.py	                                    \
            --ssl                                           \
            --sm_fn ${SM_FN}                                \
            --num_samples ${NUM_SAMP}                       \
            --lambdda ${LAMBDA}                             \
            --k_majority ${K}                               \
            --data ${DATA}                                  \
            --mdate 20230501                                \
            --epochs ${EPOCHS}                                \
            --result_epochs ${RESULT_EPOCHS}                  \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --valid_start ${VALID_START}                      \
            --valid_end ${VALID_END}                          \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --cls_feat input                                 \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --pretrined_model ${PRE}                        \
            --loss_func ${LOSS}                             \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${S}                                  \
            --bsize ${B}                                    \
            --optimizer ${OPT}                              \
            --scheduler ${SCH}                              \
            --learning_rate ${LR}                           \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs "10,500,10"                   \
            --warm_learning_rate ${WLR}                     \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --count ${CNT}                                  \
            --gpu ${GPU}                                    \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result ${RESULT_DIR}/${LOSS}/gen_apigraph_${TS}.csv \
            --log_path ${RESULT_DIR}/${LOSS}/gen_apigraph_${TS}_${EPOCHS}_${SM_FN}_${NUM_SAMP}.log \
            > ${RESULT_DIR}/${LOSS}/gen_apigraph_${TS}_${EPOCHS}_${SM_FN}_${NUM_SAMP}.log 2>&1 &