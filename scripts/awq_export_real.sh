#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=0,1,2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# python -m awq.entry \
#     --model_path "out/wanda/wanda0.0" \
#     --cache_dir "llm_weights" \
#     --w_bit "2" \
#     --q_group_size "32" \
#     --load_awq "out/awq_variants/awqw2q32/awq0.0/awq_results" \
#     --q_backend "fake" \
#     --dump_fake "out/awq_models/awqw2q32"

python arcala-prunequant/main.py \
    --model "out/awq_models/awqw2q32" \
    --prune_method "wanda" \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save "out/awq_models/awqw2q32bad" \
    --save_model "out/awq_models/awqw2q32bad" \
    --eval_seqlen 4096