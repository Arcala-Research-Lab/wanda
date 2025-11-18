# CUDA_VISIBLE_DEVICES=1 
# python ../main.py \
#     --model meta-llama/Llama-2-7b-hf \
#     --prune_method wanda \
#     --sparsity_ratio 0.5 \
#     --sparsity_type unstructured \
#     --save ../out/llama_7b/2-4/wanda/50/ \
#     --save_model ../saved_models/llama_7b/2-4/wanda_normal/50/

# echo "--- RUN 2: NEW WANDA (layerwise_scaling=True) ---"
# Note: The --layerwise_scaling flag IS present here.
python ../main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --layerwise_scaling \
    --save ../out/llama_7b/2-4/wanda_layerwise/50/ \
    --save_model ../saved_models/llama_7b/2-4/wanda_layerwise/50/

echo "--- All runs complete. Check the 'distributions' folder. ---"