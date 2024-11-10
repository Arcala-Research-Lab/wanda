#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python ../main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --save_model $5
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "0.5"  "unstructured" "../out/llama_7b/unstructured/wanda/50/" "../saved_models/llama_7b/unstructured/wanda/50/"
run_python_command "wanda" "0.5"  "2:4" "../out/llama_7b/2-4/wanda/50/" "../saved_models/llama_7b/2-4/wanda/50/"
run_python_command "wanda" "0.5"  "4:8" "../out/llama_7b/4-8/wanda/50/" "../saved_models/llama_7b/4-8/wanda/50/"
run_python_command "wanda" "0.4"  "unstructured" "../out/llama_7b/unstructured/wanda/40/" "../saved_models/llama_7b/unstructured/wanda/40/"
run_python_command "wanda" "0.4"  "2:4" "../out/llama_7b/2-4/wanda/40/" "../saved_models/llama_7b/2-4/wanda/40/"
run_python_command "wanda" "0.4"  "4:8" "../out/llama_7b/4-8/wanda/40/" "../saved_models/llama_7b/4-8/wanda/40/"
run_python_command "wanda" "0.3"  "unstructured" "../out/llama_7b/unstructured/wanda/30/" "../saved_models/llama_7b/unstructured/wanda/30/"
run_python_command "wanda" "0.3"  "2:4" "../out/llama_7b/2-4/wanda/30/" "../saved_models/llama_7b/2-4/wanda/30/"
run_python_command "wanda" "0.3"  "4:8" "../out/llama_7b/4-8/wanda/30/" "../saved_models/llama_7b/4-8/wanda/30/"
run_python_command "wanda" "0.2"  "unstructured" "../out/llama_7b/unstructured/wanda/20/" "../saved_models/llama_7b/unstructured/wanda/20/"
run_python_command "wanda" "0.2"  "2:4" "../out/llama_7b/2-4/wanda/20/" "../saved_models/llama_7b/2-4/wanda/20/"
run_python_command "wanda" "0.2"  "4:8" "../out/llama_7b/4-8/wanda/20/" "../saved_models/llama_7b/4-8/wanda/20/"
run_python_command "wanda" "0.1"  "unstructured" "../out/llama_7b/unstructured/wanda/10/" "../saved_models/llama_7b/unstructured/wanda/10/"
run_python_command "wanda" "0.1"  "2:4" "../out/llama_7b/2-4/wanda/10/" "../saved_models/llama_7b/2-4/wanda/10/"
run_python_command "wanda" "0.1"  "4:8" "../out/llama_7b/4-8/wanda/10/" "../saved_models/llama_7b/4-8/wanda/10/"
echo "Finished wanda pruning method"

# # llama-7b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
# run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/"
# run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/"
# echo "Finished magnitude pruning method"