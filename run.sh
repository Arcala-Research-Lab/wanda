#!/bin/bash

# Settings (change these as needed)
SPARSITY_RATIO=0.5
SPARSITY_TYPE="4:8"
PRUNE_METHOD="wanda"
NSAMPLES=128

# Sequence lengths to test
SEQLENS=(2048 4096)

# Model configurations: "model_path|output_name"
MODELS=(
    "/home/yichx14/llama27b/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9|llama2_7b"
    "/home/yichx14/Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b|llama31_8b"
    "/home/yichx14/models/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920|llama3_8b"
)

# Track total time
total_start=$(date +%s)

# Loop through each sequence length
for SEQLEN in "${SEQLENS[@]}"; do
    echo "########## Testing with SEQLEN=$SEQLEN ##########"
    echo ""
    
    # Loop through each model
    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r model_path model_name <<< "$model_config"
        
        echo "=========================================="
        echo "Running $model_name with seqlen=$SEQLEN..."
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Track model run time
        start=$(date +%s)
        
        python main.py \
            --model "$model_path" \
            --sparsity_ratio $SPARSITY_RATIO \
            --sparsity_type $SPARSITY_TYPE \
            --prune_method $PRUNE_METHOD \
            --seqlen $SEQLEN \
            --nsamples $NSAMPLES \
            --save "out/${model_name}_${SEQLEN}/48/wanda/max"
        
        end=$(date +%s)
        runtime=$((end - start))
        
        echo "Completed $model_name"
        echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Runtime: $((runtime / 3600))h $((runtime % 3600 / 60))m $((runtime % 60))s"
        echo "=========================================="
        echo ""
    done
done

total_end=$(date +%s)
total_runtime=$((total_end - total_start))

echo "All models and sequence lengths completed!"
echo "Total runtime: $((total_runtime / 3600))h $((total_runtime % 3600 / 60))m $((total_runtime % 60))s"