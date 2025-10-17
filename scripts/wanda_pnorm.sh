CUDA_VISIBLE_DEVICES=0 python ../main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --layerwise_scaling \
    --save out/llama_7b/unstructured/wanda/50/ \
    --save_model saved_models/llama_7b/unstructured/wanda/50/