#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_wanda () {
    python arcala-prunequant/main.py \
    --model $1 \
    --prune_method "wanda" \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --save_model $5 \
    --eval_seqlen $6 \
    > $7
}

run_wanda_new () {
    python arcala-prunequant/main.py \
    --model $1 \
    --prune_method "wanda" \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --save_model $5 \
    --eval_seqlen $6 \
    --layerwise_scaling \
    > $7
}

wanda_wrapper() {
    # ensure directories exist
    mkdir -p $4
    mkdir -p $(dirname "$6")

    # llama-7b with wanda pruning method
    echo "Running with wanda pruning method"
    run_wanda $1 $2 $3 $4 $4 $5 $6
    echo "Finished wanda pruning method"
}

wanda_new_wrapper() {
    # ensure directories exist
    mkdir -p $4
    mkdir -p $(dirname "$6")

    # llama-7b with wanda pruning method
    echo "Running with wanda pruning method"
    run_wanda_new $1 $2 $3 $4 $4 $5 $6
    echo "Finished wanda pruning method"
}

# # ======= Wanda + AWQ =======

wanda_dir="wanda"

for sparsity in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    wanda_wrapper "meta-llama/Llama-2-7b-hf" \
        $sparsity "unstructured" "out/wanda/wanda$sparsity" 4096 \
        "out/perplexities/wanda/wanda${sparsity}eval4k.txt"
done