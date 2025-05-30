#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=0,1,2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_awq () {
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --run_awq \
        --dump_awq $4
}

run_quantize () {
    mkdir -p $(dirname "$4")
    mkdir -p $(dirname "$7")

    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --load_awq $4 \
        --q_backend $5 \
        --tasks "wikitext" \
        --check_sparsity \
        --round_to_p2 \
        --eval_seqlen $6 \
        > $7
}

run_quantize_ones () {
    mkdir -p $(dirname "$4")
    mkdir -p $(dirname "$7")

    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --load_awq $4 \
        --q_backend $5 \
        --tasks "wikitext" \
        --check_sparsity \
        --set_to_1 \
        --eval_seqlen $6 \
        > $7
}

awq_pipeline() {
    # make dirs
    mkdir -p $(dirname "$4")
    mkdir -p $(dirname "$7")
    # AWQ
    echo "Running AWQ"
    run_awq $1 $2 $3 $4
    echo "Finished AWQ"
    # AWQ part 2
    echo "Running AWQ Quantization"
    run_quantize $1 $2 $3 $4 $5 $6 $7
    echo "Finished AWQ Quantization"
}

# ======= AWQ variants (normal wanda) =======

wanda_dir="wanda"

w_bits="4"
q_group_sizes="128"
sparsities="0.0"

for sparsity in $sparsities; do
    for w_bit in $w_bits; do
        for q_group_size in $q_group_sizes; do
            awq_dir="awq_2pow/awqw${w_bit}q${q_group_size}"

            run_quantize "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/awqalone/awq_results" \
                "fake" 2048 \
                "out/perplexities/$awq_dir/awq_2pow.txt"

            run_quantize_ones "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/awqalone/awq_results" \
                "fake" 2048 \
                "out/perplexities/$awq_dir/awq_1s.txt"
        done
    done
done
