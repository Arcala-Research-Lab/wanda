#!/bin/bash
# script for running wanda + awq

# Set common variables
cuda_device=0,1,2

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

run_awq () {
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size 128 \
        --run_awq \
        --dump_awq $3
}

run_quantize () {
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit $2 \
        --q_group_size 128 \
        --load_awq $3 \
        --dump_quant $4 \
        --q_backend $5
}

test_quantize() {
    python -m awq.entry \
        --model_path $1 \
        --w_bit $2 \
        --load_quant $3 \
        --cache_dir "llm_weights" \
        --q_group_size 128 \
        --q_backend $4 \
        --tasks "wikitext" \
        --eval_seqlen $5 \
    > $6
}

wanda_awq() {
    # llama-7b with wanda pruning method
    # echo "Running with wanda pruning method"
    run_wanda $1 $2 $3 $4 $4 $5 $6
    # echo "Finished wanda pruning method"
    awq_pipeline $4 $7 $8 $9 ${10} ${11} ${12}
}

awq_pipeline() {
    # # AWQ
    # echo "Running AWQ"
    # run_awq $1 $2 $3
    # echo "Finished AWQ"
    # # # AWQ part 2
    # echo "Running AWQ Quantization"
    # run_quantize $1 $2 $3 $4 $5
    # echo "Finished AWQ Quantization"
    # AWQ part 3
    echo "Evaluating AWQ Quantization"
    test_quantize $1 $2 $4 $5 $6 $7
    echo "Evaluating AWQ Quantization"
}



# # ======= AWQ alone =======

# awq_pipeline "meta-llama/Llama-2-7b-hf" 4 "out/awqalone/awq_results" "out/awqalone/quant_dump" "real" 2048 "out/perplexities/awqeval2k.txt"
# test_quantize "meta-llama/Llama-2-7b-hf" 4 "out/awqalone/quant_dump" "real" 4096 "out/perplexities/awqeval4k.txt"

# # ======= Baseline model ======
# run_wanda "meta-llama/Llama-2-7b-hf" 0 "unstructured" "out/wanda/wanda" "out/wanda/wanda" 4096 "out/perplexities/llamaeval4k"
# run_wanda "meta-llama/Llama-2-7b-hf" 0 "unstructured" "out/wanda/wanda" "out/wanda/wanda" 2048 "out/perplexities/llamaeval2k"

# # ======= Wanda + AWQ =======

# for sparsity in 0.6 0.7 0.8 0.9; do
#     wanda_awq "meta-llama/Llama-2-7b-hf" \
#         $sparsity "unstructured" "out/wanda/wanda$sparsity" 4096 \
#         "out/perplexities/wanda/wanda${sparsity}eval4k.txt" 4 \
#         "out/wanda_awq/awq${sparsity}/awq_results" \
#         "out/wanda_awq/awq${sparsity}/quant_dump" "real" 2048 \
#         "out/perplexities/wanda_awq/awq${sparsity}eval2k.txt"
# done

# ======= Wanda + AWQ eval (2k, 4k) =======

# for sparsity in 0.6 0.7 0.8 0.9; do
#     wanda_awq "out/wanda/wanda$sparsity" \
#         0 "unstructured" "out/wanda/wanda$sparsity" 2048 \
#         "out/perplexities/wanda/wanda${sparsity}eval2k.txt" 4 \
#         "out/wanda_awq/awq${sparsity}/awq_results" \
#         "out/wanda_awq/awq${sparsity}/quant_dump" "real" 4096 \
#         "out/perplexities/wanda_awq/awq${sparsity}eval4k.txt"
# done

# # # ======= Wanda + AWQ 4:8 sparsity =======
# wanda_awq "meta-llama/Llama-2-7b-hf" \
#     0.5 "4:8" "out/wanda/wanda4_8" 4096 \
#     "out/perplexities/wanda/wanda4_8eval4k.txt" 4 \
#     "out/wanda_awq/awq4_8/awq_results" \
#     "out/wanda_awq/awq4_8/quant_dump" "real" 2048 \
#     "out/perplexities/wanda_awq/awq4_8eval2k.txt"

# # ======= Wanda + AWQ 4:8 sparsity =======
# wanda_awq "out/wanda/wanda4_8" \
#     0 "unstructured" "out/wanda/wanda4_8" 2048 \
#     "out/perplexities/wanda/wanda4_8eval2k.txt" 4 \
#     "out/wanda_awq/awq4_8/awq_results" \
#     "out/wanda_awq/awq4_8/quant_dump" "real" 4096 \
#     "out/perplexities/wanda_awq/awq4_8eval4k.txt"