#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# Quit if there are any errors
set -e

for DATA_IDX in 0 1 2 3 4
do

for EXTR_LB in "w_expr" "wo_expr"
do

DATA_DIR="./data/roe-pred/$DATA_IDX/$EXTR_LB"
OUTPUT_DIR="./output/pred/$DATA_IDX/$EXTR_LB"

CUDA_VISIBLE_DEVICES=$1 python pred.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --n_hidden_layers 4 \
    --d_hidden 32 \
    --batch_size 256 \
    --n_epochs 2000 \
    --lr 0.0002 \
    --save_preds \
    --overwrite_output

done
done
