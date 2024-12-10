#!/bin/bash

# Set common variables
cuda_device=0,1,2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

awq_wanda() {
    # llama-7b with wanda pruning method
    echo "Running with wanda pruning method"
    python arcala-prunequant/main.py \
        --model "meta-llama/Llama-2-7b-hf" \
        --load_quant "out/awq/quant_dump" \
        --prune_method "wanda" \
        --sparsity_ratio $1 \
        --sparsity_type "unstructured" \
        --save $2 \
        --save_model $2 \
        --w_bit 4 \
        --q_group_size 128 \
        # > $3
    echo "Finished wanda pruning method"
}

# # ======= AWQ + Wanda =======

# awq_wanda 0.1 "out/awqwanda0.1" "out/perplexities/awqwanda0.1.txt"
# awq_wanda 0.2 "out/awqwanda0.2" "out/perplexities/awqwanda0.2.txt"
# awq_wanda 0.3 "out/awqwanda0.3" "out/perplexities/awqwanda0.3.txt"
# awq_wanda 0.4 "out/awqwanda0.4" "out/perplexities/awqwanda0.4.txt"
# awq_wanda 0.5 "out/awqwanda0.5" "out/perplexities/awqwanda0.5.txt"