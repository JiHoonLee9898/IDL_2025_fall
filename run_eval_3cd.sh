#!/bin/bash

# random, popular, adversarial
### POPE ###
CUDA_VISIBLE_DEVICES=0 python generation_3CD_pope.py \
    --model llava-1.5 \
    --model_path /datasets2/llava-1.5-7b-hf \
    --pope_type random \
    --data_path /home/donut2024/coco2014 \
    --num_images 100 \
    --seed 0 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
    --alpha 0.6 \
    --beta 1
  

# random, popular, adversarial
### POPE ###
CUDA_VISIBLE_DEVICES=0 python generation_pope.py \
    --model llava-1.5 \
    --model_path /datasets2/llava-1.5-7b-hf \
    --pope_type random \
    --data_path /home/donut2024/coco2014 \
    --num_images 100 \
    --seed 0 \
    --gpu-id 0 \
    --output_dir ./generated_captions/ \
  