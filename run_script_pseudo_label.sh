#! /bin/bash

DATA="/home/ihossain/ISMAIL/Datasets/data/gen_apigraph_drebin"
SINGLE_CKP=True
EPOCHS=240
RESULT_EPOCHS=20
K=10
BSIZE=1024
MDATE="20230501"
TRAIN_START="2012-01"
TRAIN_END="2012-12"
TEST_START="2013-01"
TEST_END="2013-01"
ENCODER="simple-enc-mlp"
ENC_HIDDEN="512-384-256-128"
MLP_HIDDEN="100-100"
LEARNING_RATE=0.003
OPTIMIZER="sgd"
CLS_FEAT="encoded"
LOSS_FUNC="hi-dist-xent"
LOG_PATH="/home/ihossain/ISMAIL/SSL-malware/pseudo_labels/pseudo_labels.log"

python -u similarity_score.py \
    --single_checkpoint ${SINGLE_CKP} \
    --epochs ${EPOCHS} \
    --result_epochs ${RESULT_EPOCHS} \
    --k_closest ${K} \
    --data ${DATA} \
    --bsize ${BSIZE} \
    --mdate ${MDATE} \
    --train_start ${TRAIN_START} \
    --train_end ${TRAIN_END} \
    --test_start ${TEST_START} \
    --test_end ${TEST_END} \
    --encoder ${ENCODER} \
    --enc_hidden ${ENC_HIDDEN} \
    --mlp_hidden ${MLP_HIDDEN} \
    --learning_rate ${LEARNING_RATE} \
    --optimizer ${OPTIMIZER} \
    --cls_feat ${CLS_FEAT} \
    --loss_func ${LOSS_FUNC} \
    --log_path ${LOG_PATH}