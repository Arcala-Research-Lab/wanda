#!/bin/bash

# Example usage:
# nohup bash test_all.sh > test_all.log 2>&1 &

# Models to test
models=(
    # "decapoda-research/llama-7b-hf"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-70B"
)

# Sparsity ratios
sparsity_ratios=(0.5)
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Sparsity types
sparsity_types=("unstructured" "2:4" "4:8")
# sparsity_types=("unstructured")

# Sequence lengths to test
seqlen_values=(2048 4096)

# Pruning method
prune_method="wanda"

# Whether to save model weights
save_model=false   # Set to false if you don't want to save models

# Function to run pruning
run_python_command () {
    local model=$1
    local prune_method=$2
    local sparsity_ratio=$3
    local sparsity_type=$4
    local use_layerwise=$5
    local seqlen=$6

    local model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]')

    # Add subfolder for layerwise vs non-layerwise runs
    local scaling_label="no_scaling"
    if [ "$use_layerwise" = true ]; then
        scaling_label="layerwise_scaling"
    fi

    local out_dir="../out/${model_name}/${sparsity_type}/${prune_method}/${sparsity_ratio}/${scaling_label}/seqlen_${seqlen}/"
    local save_model_dir="../saved_models/${model_name}/${sparsity_type}/${prune_method}/${sparsity_ratio}/${scaling_label}/seqlen_${seqlen}/"

    echo "--------------------------------------------"
    echo "Running pruning for:"
    echo " Model: $model"
    echo " Method: $prune_method"
    echo " Sparsity Ratio: $sparsity_ratio"
    echo " Sparsity Type: $sparsity_type"
    echo " Sequence Length: $seqlen"
    echo " Layerwise Scaling: $use_layerwise"
    echo " Save Model: $save_model"
    echo "--------------------------------------------"

    # Build the command dynamically
    cmd="python ../main.py \
        --model \"$model\" \
        --prune_method \"$prune_method\" \
        --sparsity_ratio \"$sparsity_ratio\" \
        --sparsity_type \"$sparsity_type\" \
        --seqlen $seqlen \
        --save \"$out_dir\""

    # Add flags as needed
    if [ "$use_layerwise" = true ]; then
        cmd+=" --layerwise_scaling"
    fi
    if [ "$save_model" = true ]; then
        cmd+=" --save_model \"$save_model_dir\""
    fi

    echo "Running command:"
    echo "$cmd"
    echo

    # Run it
    eval $cmd

    echo "Finished: $model | $prune_method | $sparsity_ratio | $sparsity_type | seqlen=$seqlen | layerwise=$use_layerwise | save_model=$save_model"
    echo
}

# Main loop
for model in "${models[@]}"; do
    for sparsity_ratio in "${sparsity_ratios[@]}"; do
        for sparsity_type in "${sparsity_types[@]}"; do
            for seqlen in "${seqlen_values[@]}"; do
                # Run both with and without layerwise scaling
                run_python_command "$model" "$prune_method" "$sparsity_ratio" "$sparsity_type" false "$seqlen"
                run_python_command "$model" "$prune_method" "$sparsity_ratio" "$sparsity_type" true "$seqlen"
            done
        done
    done
done

echo "============================================"
echo "All pruning tasks completed!"
echo "============================================"
