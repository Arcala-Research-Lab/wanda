#!/bin/bash

# Set common variables
sparsity_ratio=0.5
cuda_device=0,1,2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_wanda () {
    python arcala-prunequant/main.py \
    --model $1 \
    --prune_method $2 \
    --sparsity_ratio $3 \
    --sparsity_type $4 \
    --save $5 \
    --save_model $6 \
    > $7
}

run_awq_thresholded () {
    python -m awq.entry \
        --model_path "meta-llama/Llama-2-7b-hf" \
        --cache_dir "llm_weights" \
        --w_bit 4 \
        --q_group_size 128 \
        --run_awq \
        --threshold $1 \
        --dump_awq $2
}

run_awq () {
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit 4 \
        --q_group_size 128 \
        --run_awq \
        --dump_awq $2
}

run_quantize () {
    python -m awq.entry \
        --model_path $1 \
        --cache_dir "llm_weights" \
        --w_bit 4 \
        --q_group_size 128 \
        --load_awq $2 \
        --dump_quant $3 \
        --q_backend "real" 
}

test_quantize() {
    python -m awq.entry \
        --model_path $1 \
        --load_quant $2 \
        --cache_dir "llm_weights" \
        --w_bit 4 \
        --q_group_size 128 \
        --q_backend "real" \
        --tasks "wikitext" \
        > $3
}

test_threshold() {
    # AWQ
    echo "Running AWQ"
    run_awq_thresholded $1 $2
    echo "Finished AWQ"
    # AWQ part 2
    echo "Running AWQ Quantization"
    run_quantize "meta-llama/Llama-2-7b-hf" $2 $3
    echo "Finished AWQ Quantization"
    # AWQ part 3
    echo "Evaluating AWQ Quantization"
    test_quantize "meta-llama/Llama-2-7b-hf" $3 $4
    echo "Evaluating AWQ Quantization"
}

wanda_awq() {
    # llama-7b with wanda pruning method
    echo "Running with wanda pruning method"
    run_wanda "meta-llama/Llama-2-7b-hf" "wanda" $1  "unstructured" $2 $2 $3
    echo "Finished wanda pruning method"
    # # AWQ
    # echo "Running AWQ"
    # run_awq $2 $4
    # echo "Finished AWQ"
    # # AWQ part 2
    # echo "Running AWQ Quantization"
    # run_quantize $2 $4 $5
    # echo "Finished AWQ Quantization"
    # # AWQ part 3
    # echo "Evaluating AWQ Quantization"
    # test_quantize $2 $5 $6
    # echo "Evaluating AWQ Quantization"
}

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

#!: Control
# # AWQ
# echo "Running AWQ"
# run_awq "meta-llama/Llama-2-7b-hf" "out/awq/awq_results" 
# echo "Finished AWQ"
# # AWQ part 2
# echo "Running AWQ Quantization"
# run_quantize "meta-llama/Llama-2-7b-hf" "out/awq/awq_results" "out/awq/quant_dump"
# echo "Finished AWQ Quantization"
# # AWQ part 3
# echo "Evaluating AWQ Quantization"
# test_quantize "meta-llama/Llama-2-7b-hf" "out/awq/quant_dump" "out/perplexities/awq.txt"
# echo "Evaluating AWQ Quantization"
#!: AWQ Thresholds:
# test_threshold 3.55078125 "out/awq1threshold/awq_results" "out/awq1threshold/quant_dump" "out/perplexities/awq1threshold.txt"
# test_threshold 2.34375 "out/awq5threshold/awq_results" "out/awq5threshold/quant_dump" "out/perplexities/awq5threshold.txt"
# test_threshold 2.083984375 "out/awq10threshold/awq_results" "out/awq10threshold/quant_dump" "out/perplexities/awq10threshold.txt"
# test_threshold 1.4970703125 "out/awq20threshold/awq_results" "out/awq20threshold/quant_dump" "out/perplexities/awq20threshold.txt"
# test_threshold 1.0 "out/awq30threshold/awq_results" "out/awq30threshold/quant_dump" "out/perplexities/awq30threshold.txt"
# test_threshold 0.90771484375 "out/awq40threshold/awq_results" "out/awq40threshold/quant_dump" "out/perplexities/awq40threshold.txt"
# test_threshold 0.884765625 "out/awq50threshold/awq_results" "out/awq50threshold/quant_dump" "out/perplexities/awq50threshold.txt"
# test_threshold 0.84814453125 "out/awq60threshold/awq_results" "out/awq60threshold/quant_dump" "out/perplexities/awq60threshold.txt"
# test_threshold 0.8173828125 "out/awq70threshold/awq_results" "out/awq70threshold/quant_dump" "out/perplexities/awq70threshold.txt"
# test_threshold 0.7841796875 "out/awq80threshold/awq_results" "out/awq80threshold/quant_dump" "out/perplexities/awq80threshold.txt"
# test_threshold 0.72314453125 "out/awq90threshold/awq_results" "out/awq90threshold/quant_dump" "out/perplexities/awq90threshold.txt"
# control threshold:
# test_threshold inf "out/awqnoscale/awq_results" "out/awqnoscale/quant_dump" "out/perplexities/awqnoscale.txt"
#!: Wanda + AWQ:
# wanda_awq 0.1 "out/wanda0.1" "out/perplexities/wanda0.1.txt" "out/awq0.1/awq_results" "out/awq0.1/quant_dump" "out/perplexities/awq0.1.txt"
# wanda_awq 0.2 "out/wanda0.2" "out/perplexities/wanda0.2.txt" "out/awq0.2/awq_results" "out/awq0.2/quant_dump" "out/perplexities/awq0.2.txt"
# wanda_awq 0.3 "out/wanda0.3" "out/perplexities/wanda0.3.txt" "out/awq0.3/awq_results" "out/awq0.3/quant_dump" "out/perplexities/awq0.3.txt"
# wanda_awq 0.4 "out/wanda0.4" "out/perplexities/wanda0.4.txt" "out/awq0.4/awq_results" "out/awq0.4/quant_dump" "out/perplexities/awq0.4.txt"
# wanda_awq 0.5 "out/wanda0.5" "out/perplexities/wanda0.5.txt" "out/awq0.5/awq_results" "out/awq0.5/quant_dump" "out/perplexities/awq0.5.txt"
#!: AWQ + Wanda:
awq_wanda 0.1 "out/awqwanda0.1" "out/perplexities/awqwanda0.1.txt"
# awq_wanda 0.2 "out/awqwanda0.2" "out/perplexities/awqwanda0.2.txt"
# awq_wanda 0.3 "out/awqwanda0.3" "out/perplexities/awqwanda0.3.txt"
# awq_wanda 0.4 "out/awqwanda0.4" "out/perplexities/awqwanda0.4.txt"
# awq_wanda 0.5 "out/awqwanda0.5" "out/perplexities/awqwanda0.5.txt"