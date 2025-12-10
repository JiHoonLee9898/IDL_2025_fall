#!/bin/bash

# Test script for MiniGPT4 model (LLaMA2)
# Configuration: eval_configs/minigpt4_llama2_eval.yaml

# Default parameters (modify as needed)
MODEL="mini_gpt4_llama2"
MODEL_PATH="${MODEL_PATH:-/home/donut2024/RVCD/MAIN_CODES/prerained_minigpt4_7b.pth}"
DATA_PATH="${DATA_PATH:-/home/donut2024/coco2014}"
GPU_ID="${GPU_ID:-0}"
POPE_TYPE="${POPE_TYPE:-random}"  # Options: random, popular, adversarial
NUM_IMAGES="${NUM_IMAGES:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-./paper_result/}"
GT_SEG_PATH="${GT_SEG_PATH:-pope_coco/coco_ground_truth_segmentation.json}"

echo "Running MiniGPT4 (LLaMA2) POPE evaluation..."
echo "Model: $MODEL"
echo "POPE Type: $POPE_TYPE"
echo "GPU: $GPU_ID"

python generate_test.py \
    --model "$MODEL" \
    --model_path "$MODEL_PATH" \
    --pope_type "$POPE_TYPE" \
    --gpu-id "$GPU_ID" \
    --data_path "$DATA_PATH" \
    --num_images "$NUM_IMAGES" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --gt_seg_path "$GT_SEG_PATH"

echo "Test completed!"
