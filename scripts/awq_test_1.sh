export CUDA_VISIBLE_DEVICES=1

q_group_sizes="512"

for q_group_size in $q_group_sizes; do
    python -m awq.entry \
        --model_path "meta-llama/Llama-2-7b-hf" \
        --cache_dir "llm_weights" \
        --w_bit 2 \
        --q_group_size ${q_group_size} \
        --run_awq \
        --dump_awq "out/awq_results${q_group_size}"

    python -m awq.entry \
        --model_path "meta-llama/Llama-2-7b-hf" \
        --cache_dir "llm_weights" \
        --w_bit 2 \
        --q_group_size ${q_group_size} \
        --run_awq \
        --dump_awq "out/awq_results${q_group_size}4"

    python -m awq.entry \
        --model_path "meta-llama/Llama-2-7b-hf" \
        --cache_dir "llm_weights" \
        --w_bit 2 \
        --q_group_size ${q_group_size} \
        --run_awq \
        --load_awq "out/awq_results${q_group_size}" \
        --prune_highbit \
        --q_backend "fake" \
        --tasks "wikitext" \
        --check_sparsity \
        --eval_seqlen "2048" \
        > "out/perplex${q_group_size}.txt"
done



