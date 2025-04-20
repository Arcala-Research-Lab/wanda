#!/bin/bash

# IMPORTANT: MAKE SURE TO SET DEFAULT ARG in ppl_eval.py FOR ctx_length TO 4096 OR 2048
# OBTAIN PPL FOR PRUNED BASE MODELS AND LORA FT MODELS - for models pruned with the improved technique (Ask Omar)

echo "10% sparsity" > ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.1 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.1 --lora_weights ft_pruned_models/wanda_improved/wanda0.1 >> ppl_logs/wanda_improved_all.log

echo "20% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.2 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.2 --lora_weights ft_pruned_models/wanda_improved/wanda0.2 >> ppl_logs/wanda_improved_all.log

echo "30% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.3 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.3 --lora_weights ft_pruned_models/wanda_improved/wanda0.3 >> ppl_logs/wanda_improved_all.log

echo "40% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.4 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.4 --lora_weights ft_pruned_models/wanda_improved/wanda0.4 >> ppl_logs/wanda_improved_all.log

echo "50% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.5 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.5 --lora_weights ft_pruned_models/wanda_improved/wanda0.5 >> ppl_logs/wanda_improved_all.log

echo "60% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.6 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.6 --lora_weights ft_pruned_models/wanda_improved/wanda0.6 >> ppl_logs/wanda_improved_all.log

echo "70% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.7 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.7 --lora_weights ft_pruned_models/wanda_improved/wanda0.7 >> ppl_logs/wanda_improved_all.log

echo "80% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.8 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.8 --lora_weights ft_pruned_models/wanda_improved/wanda0.8 >> ppl_logs/wanda_improved_all.log

echo "90% sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.9 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda0.9 --lora_weights ft_pruned_models/wanda_improved/wanda0.9 >> ppl_logs/wanda_improved_all.log

echo "2:4 sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda2_4 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda2_4 --lora_weights ft_pruned_models/wanda_improved/wanda2_4 >> ppl_logs/wanda_improved_all.log

echo "4:8 sparsity" >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda4_8 >> ppl_logs/wanda_improved_all.log
python ppl_eval.py --model improved_models/wanda4_8 --lora_weights ft_pruned_models/wanda_improved/wanda4_8 >> ppl_logs/wanda_improved_all.log