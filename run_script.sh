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
DATA=/home/mhaque3/myDir/data/gen_androzoo_drebin
TRAIN_START=2019-01
TRAIN_END=2019-12
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
            --log_path ${RESULT_DIR}/gen_apigraph_cnt${CNT}_${TS}.log \
            > ${RESULT_DIR}/gen_apigraph_cnt${CNT}_${TS}.log 2>&1