#!/bin/bash

# Define ranges for weight_power and wanda_power
# weight_power_start=0.05
# weight_power_end=0.95
awq_power_start=0.05
awq_power_end=2
step=0.05

# Define layer names
layer_names=(
    "mlp.gate_proj"
    "mlp.up_proj"
    "self_attn.v_proj"
    "mlp.down_proj"
    "self_attn.q_proj"
    "self_attn.k_proj"
    "self_attn.o_proj"
    # "all"
)

# Create a directory for logs if it doesn't exist
mkdir -p compare_logs

# Log file path
log_file="compare_logs/wanda_new10_50_layers_awq.log"

# Write header to the log file
echo "Starting grid test for wanda_power, weight_power, and layer_name" > $log_file
echo "Log format: weight_power=VALUE, wanda_power=VALUE, layer_name=VALUE" >> $log_file
echo "===============================================" >> $log_file

# Loop over weight_power, wanda_power, and layer_name
for layer_name in "${layer_names[@]}"; do
    for awq_power in $(seq $awq_power_start $step $awq_power_end); do
        # Log the current combination
        echo "Running test: weight_power=1.0, awq_power=$awq_power, layer_name=$layer_name" | tee -a $log_file

        # Run the Python script with the specified parameters
        python main.py \
            --model meta-llama/Llama-2-7b-hf \
            --prune_method wanda_new \
            --mode 10 \
            --sparsity_ratio 0.5 \
            --sparsity_type unstructured \
            --weight_power 1.0 \
            --awq_power $awq_power \
            --layer_name $layer_name \
            >> $log_file 2>&1

        # Log separator after each test
        echo "-----------------------------------------------" >> $log_file
    done
done

echo "Grid testing completed. Logs saved to $log_file"