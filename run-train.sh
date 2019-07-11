#!/bin/bash

CUDA_CACHE_PATH=/st2/hayeon/tmp python3 train.py \
    --dataset tiered \
    --num_ways 5 \
    --num_shots 5 \
    --transductive True \
    --gpu 2,4,5,6,7 \
    --meta_batch_size 40 \
