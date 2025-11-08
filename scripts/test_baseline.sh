#!/bin/bash

# Example usage:
# nohup bash test_baseline.sh > test_baseline.log 2>&1 &

# =============================
#     MODELS TO TEST
# =============================
models=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-70B"
    # "meta-llama/Llama-4-Scout-17B-16E"
)

# =============================
#   SEQUENCE LENGTHS TO TEST
# =============================
seqlen_values=(2048 4096)


# =============================
#   BASELINE EVALUATION FUNC
# =============================
run_baseline() {
    local model=$1
    local seqlen=$2
    local model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]')
    local out_dir="../out/${model_name}/baseline/seqlen_${seqlen}/"
    mkdir -p "$out_dir"

    echo "--------------------------------------------"
    echo "Running baseline perplexity check for:"
    echo " Model: $model"
    echo " Sequence Length: $seqlen"
    echo "--------------------------------------------"


    cmd="python ../main.py \
        --model \"$model\" \
        --sparsity_ratio 0.0 \
        --sparsity_type baseline \
        --prune_method baseline \
        --seqlen $seqlen \
        --save \"$out_dir\""

    echo "Running command:"
    echo "$cmd"
    echo

    eval $cmd

    echo "Finished baseline PPL for model: $model (seqlen=$seqlen)"
    echo "Results saved to: $out_dir"
    echo
}

# =============================
#   MAIN LOOP
# =============================
for model in "${models[@]}"; do
    for seqlen in "${seqlen_values[@]}"; do
        run_baseline "$model" "$seqlen"
    done
done

echo "============================================"
echo "All baseline PPL evaluations completed!"
echo "============================================"
