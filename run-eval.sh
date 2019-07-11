#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 CUDA_CACHE_PATH=/st2/hayeon/tmp python3 eval.py --test_model D-mini_N-5_K-5_U-0_L-3_B-40_T-False --num_gpus 2
