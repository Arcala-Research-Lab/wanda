#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    /home/oyahia/.conda/envs/RLPruner/bin/python wanda/main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --save_model $5 \
    --eval_seqlen 4096 \
    --nsamples 32
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
# > "/srv/disk00/oyahia/out/wanda_test/wanda0.6ppl"
run_python_command "wanda_optimized" "0.6" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda0.6" "/srv/disk00/oyahia/out/wanda_test/wanda0.6" > "/srv/disk00/oyahia/out/wanda_test/wanda0.6ppl"
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