# python arcala-prunequant/main.py \
#     --model "meta-llama/Llama-2-7b-hf" \
#     --sparsity_type "unstructured"  \
#     --sparsity_ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
#     --quantiles 0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 \
#     --save_masks "out/masks" \
#     --calculate_masks

python arcala-prunequant/main.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --sparsity_type "unstructured"  \
    --sparsity_ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --quantiles 0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 \
    --save_masks "out/masks" \
    --save_comparisons "out/comparisons" \
    --compare_selection
