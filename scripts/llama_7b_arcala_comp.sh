#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
# model="facebook/opt-6.7b"
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python /pub/oyahia/arcala-prunequant/main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratios $2 \
    --sparsity_type $3 \
    --thresholds $4 \
    --save $5 \
    --save_model $6 \
    --compare_selection
}

# llama-7b with wanda pruning method
echo "Starting shell script"
run_python_command "wanda" "0.1 0.2 0.3 0.4 0.5" "unstructured" "3.55078125 2.34375 2.083984375" "../out/llama_7b/unstructured/wanda/30/" "../saved_models/llama_7b/unstructured/wanda/30/"
echo "Finished shell script"