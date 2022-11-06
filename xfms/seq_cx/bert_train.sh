#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run bert_train.py
# -------------------------------------

# Quit if there are any errors
set -e

TRAIN_DIR="../data/train-fp/train.json"
TEST_DIR="../data/train-fp/train.json"
OUTPUT_DIR="./biobert-classifier"
BERT_MODEL="dmis-lab/biobert-v1.1"
NUM_TRAIN_EPOCHS=80
VALID_TOLERENCE=20
MAX_LENGTH=256
BATCH_SIZE=16
SEED=0
LR=2e-6


CUDA_VISIBLE_DEVICES=$1 python bert_train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_file $TRAIN_DIR \
    --test_file $TEST_DIR \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --num_valid_tolerance $VALID_TOLERENCE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --seed $SEED
