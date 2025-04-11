#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_quantize_no_dump () {
    mkdir -p $(dirname "$4")
    mkdir -p $(dirname "$6")
    mkdir -p $(dirname "$8")

    /home/oyahia/.conda/envs/arcala_prunequant/bin/python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --load_awq $4 \
        --q_backend $5 \
        --tasks "wikitext" \
        --check_sparsity \
        --eval_seqlen $7 \
        > $8
}

run_quantize () {
    mkdir -p $(dirname "$4")
    mkdir -p $(dirname "$6")
    mkdir -p $(dirname "$8")

    /home/oyahia/.conda/envs/arcala_prunequant/bin/python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --load_awq $4 \
        --q_backend $5 \
        --dump_fake $6 \
        --tasks "wikitext" \
        --check_sparsity \
        --eval_seqlen $7 \
        > $8
}

run_wanda () {
    /home/oyahia/.conda/envs/arcala_prunequant/bin/python wanda/main.py \
    --model $1 \
    --cache_dir "llm_weights" \
    --prune_method "wanda" \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --eval_seqlen $5 \
    --layerwise_scaling 
}

# ======= Get 0.0 Sparsity Models =======

# wanda_dir="wanda"

# w_bits="2 3 4"
# q_group_sizes="2 4 8 16 32 64 128"
# sparsities="0.0"

# for sparsity in $sparsities; do
#     for w_bit in $w_bits; do
#         for q_group_size in $q_group_sizes; do
#             out_dir="/srv/disk00/oyahia/out"
#             awq_dir="awq_variants/awqw${w_bit}q${q_group_size}"

#             # run_quantize "meta-llama/Llama-2-7b-hf" $w_bit $q_group_size \
#             #     "$out_dir/$awq_dir/awq${sparsity}/awq_results" \
#             #     "fake" "$out_dir/$awq_dir/awq${sparsity}/awq_model" 2048 \
#             #     "$out_dir/perplexities/$awq_dir/awq${sparsity}eval2k.txt"

            
#             run_quantize_no_dump "meta-llama/Llama-2-7b-hf" $w_bit $q_group_size \
#                 "$out_dir/$awq_dir/awq${sparsity}/awq_results" \
#                 "fake" "$out_dir/$awq_dir/awq${sparsity}/awq_model" 4096 \
#                 "$out_dir/perplexities/$awq_dir/awq${sparsity}eval4k.txt"
#         done
#     done
# done

# # ======= AWQ variants (latest wanda) =======


# w_bits="2 3 4"
# q_group_sizes="2 4 8 16 32 64 128"
# sparsities="0.5 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9"

# for sparsity in $sparsities; do
#     for w_bit in $w_bits; do
#         for q_group_size in $q_group_sizes; do
#             out_dir="/srv/disk00/oyahia/out"
#             awq_dir="awq_variants/awqw${w_bit}q${q_group_size}"

#             run_wanda "$out_dir/$awq_dir/awq0.0/awq_model" $sparsity "unstructured" \
#                 "$out_dir/$awq_dir/awq${sparsity}" 2048 \
#                 "$out_dir/perplexities/$awq_dir/awq${sparsity}eval2k.txt"
#             run_wanda "$out_dir/$awq_dir/awq0.0/awq_model" $sparsity "unstructured" \
#                 "$out_dir/$awq_dir/awq${sparsity}" 4096 \
#                 "$out_dir/perplexities/$awq_dir/awq${sparsity}eval4k.txt"
#         done
#     done
# done

run_wanda "meta-llama/Llama-3.1-8B" 0.5 "unstructured" \
    "/home/oyahia/yeh-research" 2048