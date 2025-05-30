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
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size $3 \
        --load_awq $4 \
        --no_zero_point \
        --q_backend $5 \
        --tasks "wikitext" \
        --check_sparsity \
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

wanda_dir="wanda_wmetric_layered"

w_bits="2"
q_group_sizes="32"
sparsities="0.5"

for sparsity in $sparsities; do
    for w_bit in $w_bits; do
        for q_group_size in $q_group_sizes; do
            if [[ $w_bit == "4" && $q_group_size == "128" ]]; then
                echo "Skipping w=4,q=128 b/c already tested"
                continue
            fi

            awq_dir="awq_variants_latest/awqw${w_bit}q${q_group_size}"
            awq_dir_out="awq_variants_test/awqw${w_bit}q${q_group_size}"

            run_quantize "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/$awq_dir/awq${sparsity}/awq_results" \
                "fake" 2048 \
                "out/perplexities/$awq_dir_out/awq${sparsity}eval2k_nozeropoint.txt"
        done
    done
done