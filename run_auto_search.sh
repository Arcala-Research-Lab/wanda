#!/bin/bash

# Example script to run WANDA pruning with automatic hyperparameter search
# This script demonstrates how to use the new auto-search functionality

# Basic parameters
MODEL="meta-llama/Llama-2-7b-hf"
SPARSITY_RATIO="0.5"
LAYER_NAME="all"  # Target all layers for optimization

# Auto-search parameters
GRID_SIZE=20  # 10x10 grid search (100 combinations)
WEIGHT_POWER_MIN=0.10
WEIGHT_POWER_MAX=2.0
WANDA_POWER_MIN=0.10
WANDA_POWER_MAX=2.0

# Output directory
OUTPUT_DIR="out/llama2_7b_auto_search"

echo "Running WANDA with automatic hyperparameter search..."
echo "Model: $MODEL"
echo "Sparsity: $SPARSITY_RATIO"
echo "Target layer: $LAYER_NAME"
echo "Search grid: ${GRID_SIZE}x${GRID_SIZE}"
echo "Output directory: $OUTPUT_DIR"

python main.py \
    --model $MODEL \
    --prune_method wanda_auto \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type unstructured \
    --auto_search \
    --optimize_per_layer \
    --layer_name $LAYER_NAME \
    --search_grid_size $GRID_SIZE \
    --weight_power_min $WEIGHT_POWER_MIN \
    --weight_power_max $WEIGHT_POWER_MAX \
    --wanda_power_min $WANDA_POWER_MIN \
    --wanda_power_max $WANDA_POWER_MAX \
    --save $OUTPUT_DIR \
    --nsamples 128 \
    --seed 0

echo "Auto-search completed. Results saved to $OUTPUT_DIR"
echo "Optimal parameters saved to $OUTPUT_DIR/optimal_parameters.json"