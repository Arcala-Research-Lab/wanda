#!/bin/bash

# Conduct all levels of pruning supported by Wanda on Llama 2

python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.1 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_1 > prune_sweep_logs/wanda/prune_sweeps_0_1.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.2 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_2 > prune_sweep_logs/wanda/prune_sweeps_0_2.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.3 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_3 > prune_sweep_logs/wanda/prune_sweeps_0_3.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.4 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_4 > prune_sweep_logs/wanda/prune_sweeps_0_4.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_5 > prune_sweep_logs/wanda/prune_sweeps_0_5.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_6 > prune_sweep_logs/wanda/prune_sweeps_0_6.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.7 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_7 > prune_sweep_logs/wanda/prune_sweeps_0_7.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.8 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_8 > prune_sweep_logs/wanda/prune_sweeps_0_8.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.9 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/ --save_model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_9 > prune_sweep_logs/wanda/prune_sweeps_0_9.log

python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type 2:4 --save out/llama_7b/structured/wanda/ --save_model out/llama_7b/structured/wanda/pruned_models/llama_7b_2_4 > prune_sweep_logs/wanda/prune_sweeps_2_4.log
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type 4:8 --save out/llama_7b/structured/wanda/ --save_model out/llama_7b/structured/wanda/pruned_models/llama_7b_4_8 > prune_sweep_logs/wanda/prune_sweeps_4_8.log
