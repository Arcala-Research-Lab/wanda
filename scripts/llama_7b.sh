#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=0,1

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
    --eval_seqlen $6
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
    --layerwise_scaling
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

# for sparsity in 0.5; do
#     wanda_wrapper "meta-llama/Llama-2-7b-hf" \
#         $sparsity "unstructured" "out/wanda_wikitext/wanda$sparsity" 4096 \
#         "out/perplexities/wanda_c4/wanda${sparsity}eval4k.txt"
# done

# c4:
# 8.945090293884277
# 8.852150917053223
# 8.845528602600098

# wikitext2:
# 6.306099891662598
# 6.265037536621094
# 6.274524688720703
# 6.272838592529297
# 6.3286356925964355

# 0.4917:
# 6.309859275817871

for sparsity in 0.5; do
    wanda_new_wrapper "meta-llama/Llama-2-7b-hf" \
        $sparsity "unstructured" "out/wanda_latest_c4/wanda$sparsity" 4096 \
        "out/perplexities/wanda_latest_wikitext/wanda${sparsity}eval4k.txt"
done