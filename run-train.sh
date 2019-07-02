#!/bin/bash

CUDA_VISIBLE_DEVICES=1 CUDA_CACHE_PATH=/st2/hayeon/tmp python3 train.py --dataset mini --num_ways 5 --num_shots 5 --transductive True
