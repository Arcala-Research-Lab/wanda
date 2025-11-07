#!/bin/bash

# Test script for per-layer auto-search with smaller grid for faster testing
# This will optimize weight_power and wanda_power for each layer individually

# Basic parameters
MODEL="meta-llama/Llama-2-7b-hf"
SPARSITY_RATIO="0.5"

# Auto-search parameters (smaller grid for testing)
GRID_SIZE=10  # 10x10 grid search (100 combinations per layer)
WEIGHT_POWER_MIN=0.0
WEIGHT_POWER_MAX=2.0
WANDA_POWER_MIN=0.0
WANDA_POWER_MAX=2.0

# Output directory
OUTPUT_DIR="out/llama2_7b_per_layer_test"

echo "Running WANDA with per-layer automatic hyperparameter search..."
echo "Model: $MODEL"
echo "Sparsity: $SPARSITY_RATIO"
echo "Search grid: ${GRID_SIZE}x${GRID_SIZE} per layer"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will:"
echo "1. Find optimal weight_power and wanda_power using PERPLEXITY optimization"
echo "2. Apply all optimal parameters and measure final perplexity"
echo "3. Save detailed results and optimal parameters"

mkdir -p $OUTPUT_DIR

python main.py \
    --model $MODEL \
    --eval_seqlen 2048 \
    --prune_method wanda_auto \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type unstructured \
    --auto_search \
    --optimize_per_layer \
    --layer_name all \
    --search_grid_size $GRID_SIZE \
    --weight_power_min $WEIGHT_POWER_MIN \
    --weight_power_max $WEIGHT_POWER_MAX \
    --wanda_power_min $WANDA_POWER_MIN \
    --wanda_power_max $WANDA_POWER_MAX \
    --optimization_metric perplexity \
    --save $OUTPUT_DIR \
    --nsamples 64 \
    --seed 0 \
    2>&1 | tee $OUTPUT_DIR/run_log.txt

echo ""
echo "Per-layer auto-search completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Run log: $OUTPUT_DIR/run_log.txt"
echo "Optimal parameters: $OUTPUT_DIR/optimal_parameters.json"