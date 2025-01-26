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
        --q_backend $5 \
        --tasks "wikitext" \
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

w_bits="2 3 4"
# q_group_sizes="1 2 4 8 16 32 64 128"
q_group_sizes="1 2 4"
sparsities="0.5 2_4 4_8 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9"

for sparsity in $sparsities; do
    for w_bit in $w_bits; do
        for q_group_size in $q_group_sizes; do
            if [[ $w_bit == "4" && $q_group_size == "128" ]]; then
                echo "Skipping w=4,q=128 b/c already tested"
                continue
            fi

            awq_dir="awq_variants/awqw${w_bit}q${q_group_size}"

            awq_pipeline "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/$awq_dir/awq${sparsity}/awq_results" \
                "fake" 2048 \
                "out/perplexities/$awq_dir/awq${sparsity}eval2k.txt"

            run_quantize "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/$awq_dir/awq${sparsity}/awq_results" \
                "fake" 4096 \
                "out/perplexities/$awq_dir/awq${sparsity}eval4k.txt"
        done
    done
done

# ======= AWQ variants (latest wanda) =======

wanda_dir="wanda_wmetric_layered"

w_bits="2 3 4"
# q_group_sizes="1 2 4 8 16 32 64 128"
q_group_sizes="1 2 4"
sparsities="0.5 2_4 4_8 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9"

for sparsity in $sparsities; do
    for w_bit in $w_bits; do
        for q_group_size in $q_group_sizes; do
            if [[ $w_bit == "4" && $q_group_size == "128" ]]; then
                echo "Skipping w=4,q=128 b/c already tested"
                continue
            fi

            awq_dir="awq_variants_latest/awqw${w_bit}q${q_group_size}"

            awq_pipeline "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/$awq_dir/awq${sparsity}/awq_results" \
                "fake" 2048 \
                "out/perplexities/$awq_dir/awq${sparsity}eval2k.txt"

            run_quantize "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
                "out/$awq_dir/awq${sparsity}/awq_results" \
                "fake" 4096 \
                "out/perplexities/$awq_dir/awq${sparsity}eval4k.txt"
        done
    done
done

# ======= AWQ sanity check =======

# wanda_dir="wanda_wmetric_layered"

# w_bits="3"
# q_group_sizes="16"
# sparsities="0.8"

# for sparsity in $sparsities; do
#     for w_bit in $w_bits; do
#         for q_group_size in $q_group_sizes; do
#             if [[ $w_bit == "4" && $q_group_size == "128" ]]; then
#                 echo "Skipping w=4,q=128 b/c already tested"
#                 continue
#             fi

#             awq_dir="awq_variants_latest/awqw${w_bit}q${q_group_size}"

#             run_quantize "out/$wanda_dir/wanda$sparsity" $w_bit $q_group_size \
#                 "out/$awq_dir/awq${sparsity}/awq_results" \
#                 "fake" 4096 \
#                 "out/perplexities/$awq_dir/awq${sparsity}eval4k.txt"
#         done
#     done
# done
