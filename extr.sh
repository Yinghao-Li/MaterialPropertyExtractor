#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# Quit if there are any errors
set -e

RAW_ARTICLE_DIR="./data/gas_permeability_dataset"
PROCESSED_ARTICLE_DIR="./data/gas_permeability_dataset_processed"

for PROPERTY_NAME in "o2_permeability" "co2_permeability"
do

EXTR_RESULT_DIR="./results/IE-$PROPERTY_NAME-v2"
KEYWORD_PATH="./dependency/$PROPERTY_NAME.json"
BATCH_SIZE=128

CUDA_VISIBLE_DEVICES=$1 python extr.py \
    --raw_article_dir $RAW_ARTICLE_DIR \
    --processed_article_dir $PROCESSED_ARTICLE_DIR \
    --extr_result_dir $EXTR_RESULT_DIR \
    --keyword_path $KEYWORD_PATH \
    --batch_size $BATCH_SIZE \
    --do_parsing \
    --do_extraction \
    --save_html \
    --save_jsonl

done
