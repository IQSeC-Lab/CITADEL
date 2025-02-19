#! /bin/bash

SEQ=088
LR=0.003
OPT=sgd
SCH=step
DECAY=0.95
E=250
WLR=0.00015
WE=100
DATA=/home/mhaque3/myDir/data/gen_apigraph_drebin
#DATA=/home/mhaque3/myDir/data/gen_androzoo_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=results_ours
AL_OPT=adam

CNT=200

modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist-xent'
TS=$(date "+%m.%d-%H.%M.%S")

python -u main.py	                                \
            --data ${DATA}                                  \
            --mdate 20230501                                \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --bsize ${B}                                    \
            --epochs ${E}                                   \
            --sampler ${S}                                  \
            --optimizer ${OPT}                              \
            --learning_rate ${LR}                           \
            --scheduler ${SCH}                              \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs [10,500,10]                   \
            --xent-lambda 100                               \
            --loss_func ${LOSS}                             \
            --warm_learning_rate ${WLR}                     \
            --display-interval 180                          \
            --count ${CNT}                                  \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result ${RESULT_DIR}/gen_apigraph_cnt${CNT}_${TS}.csv \
            --log_path ${RESULT_DIR}/gen_apigraph_cnt${CNT}_${TS}.log \
            > ${RESULT_DIR}/gen_apigraph_cnt${CNT}_${TS}.log 2>&1