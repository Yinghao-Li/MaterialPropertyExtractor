#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# Quit if there are any errors
set -e

RAW_ARTICLE_DIR="./data"
PROCESSED_ARTICLE_DIR="./data_processed"
EXTR_RESULT_DIR="./results/"
KEYWORD_PATH="./dependency/keywords.json"
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
