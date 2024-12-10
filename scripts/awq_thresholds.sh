#!/bin/bash

# Set common variables
cuda_device=0,1,2

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

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

# # ======= AWQ Thresholds (1,5,10,20,30,40,50,60,70,80,90,100) =======

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