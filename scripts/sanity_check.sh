run_wanda () {
    python arcala-prunequant/main.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --prune_method "wanda" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --eval_seqlen 4096 \
    --capture_scaler_row
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
        --q_backend $4 \
        --tasks "wikitext" \
        --eval_seqlen $5 \
        --check_sparsity 
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
        --no_zero_point \
        --check_sparsity
}

awq_pipeline() {
    # AWQ
    echo "Running AWQ"
    run_awq $1 $2 $3
    echo "Finished AWQ"
    # AWQ part 2
    echo "Running AWQ Quantization"
    run_quantize $1 $2 $3 $4 $5
    echo "Finished AWQ Quantization"
    # AWQ part 3
    echo "Evaluating AWQ Quantization"
    test_quantize $1 $2 $4 $5 $6 $7
    echo "Evaluating AWQ Quantization"
}

run_wanda