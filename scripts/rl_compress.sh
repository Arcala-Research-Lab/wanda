python wanda/rl_compress.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --save "out/test" \
    --save_model "out/test" \
    --eval_seqlen 4096