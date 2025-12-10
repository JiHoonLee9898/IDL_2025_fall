#!/bin/bash

# Run tests for all POPE types (random, popular, adversarial)
# for a given model

# Usage: ./run_all_pope_types.sh <model_script>
# Example: ./run_all_pope_types.sh run_llava_1.5_test.sh

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_test_script>"
    echo "Example: $0 run_llava_1.5_test.sh"
    exit 1
fi

MODEL_SCRIPT="$1"

if [ ! -f "$MODEL_SCRIPT" ]; then
    echo "Error: Script $MODEL_SCRIPT not found!"
    exit 1
fi

echo "Running all POPE types for $MODEL_SCRIPT..."
echo "=========================================="

# Run random POPE
echo ""
echo "1/3 - Running RANDOM POPE evaluation..."
POPE_TYPE="random" bash "$MODEL_SCRIPT"

# Run popular POPE
echo ""
echo "2/3 - Running POPULAR POPE evaluation..."
POPE_TYPE="popular" bash "$MODEL_SCRIPT"

# Run adversarial POPE
echo ""
echo "3/3 - Running ADVERSARIAL POPE evaluation..."
POPE_TYPE="adversarial" bash "$MODEL_SCRIPT"

echo ""
echo "=========================================="
echo "All POPE evaluations completed!"
