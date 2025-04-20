#!/bin/bash

# IMPORTANT: MAKE SURE TO SET DEFAULT ARG in ppl_eval.py FOR ctx_length TO 4096 OR 2048
# OBTAIN PPL FOR PRUNED BASE MODELS AND LORA FT MODELS - for models pruned by wanda normally

echo "0% sparsity" > ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_0/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_0/ --lora_weights ft_pruned_models/wanda/llama_7b_0_0 >> ppl_logs/wanda_original_all.log

echo "10% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_1/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_1/ --lora_weights ft_pruned_models/wanda/llama_7b_0_1 >> ppl_logs/wanda_original_all.log

echo "20% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_2/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_2/ --lora_weights ft_pruned_models/wanda/llama_7b_0_2 >> ppl_logs/wanda_original_all.log

echo "30% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_3/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_3/ --lora_weights ft_pruned_models/wanda/llama_7b_0_3 >> ppl_logs/wanda_original_all.log

echo "40% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_4/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_4/ --lora_weights ft_pruned_models/wanda/llama_7b_0_4 >> ppl_logs/wanda_original_all.log

echo "50% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_5/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_5/ --lora_weights ft_pruned_models/wanda/llama_7b_0_5 >> ppl_logs/wanda_original_all.log

echo "60% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_6/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_6/ --lora_weights ft_pruned_models/wanda/llama_7b_0_6 >> ppl_logs/wanda_original_all.log

echo "70% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_7/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_7/ --lora_weights ft_pruned_models/wanda/llama_7b_0_7 >> ppl_logs/wanda_original_all.log

echo "80% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_8/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_8/ --lora_weights ft_pruned_models/wanda/llama_7b_0_8 >> ppl_logs/wanda_original_all.log

echo "90% sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_9/ >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/unstructured/wanda/pruned_models/llama_7b_0_9/ --lora_weights ft_pruned_models/wanda/llama_7b_0_9 >> ppl_logs/wanda_original_all.log

echo "2:4 sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/structured/wanda/pruned_models/llama_7b_2_4 >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/structured/wanda/pruned_models/llama_7b_2_4 --lora_weights ft_pruned_models/wanda/llama_7b_2_4 >> ppl_logs/wanda_original_all.log

echo "4:8 sparsity" >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/structured/wanda/pruned_models/llama_7b_4_8 >> ppl_logs/wanda_original_all.log
python ppl_eval.py --model out/llama_7b/structured/wanda/pruned_models/llama_7b_4_8 --lora_weights ft_pruned_models/wanda/llama_7b_4_8 >> ppl_logs/wanda_original_all.log