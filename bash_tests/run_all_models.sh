#!/bin/bash

# Run tests for all available models with a specific POPE type

POPE_TYPE="${POPE_TYPE:-random}"  # Default to random, can be overridden

echo "Running all models with POPE type: $POPE_TYPE"
echo "=============================================="

# Array of test scripts
SCRIPTS=(
    "run_llava_1.5_test.sh"
    "run_minigpt4_test.sh"
    "run_minigpt4_llama2_test.sh"
    "run_mplug_owl2_test.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo ""
        echo "Running $script..."
        echo "----------------------------------------"
        POPE_TYPE="$POPE_TYPE" bash "$script"
    else
        echo "Warning: $script not found, skipping..."
    fi
done

echo ""
echo "=============================================="
echo "All model evaluations completed!"
