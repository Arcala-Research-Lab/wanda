#!/bin/bash

# Define ranges for weight_power and wanda_power
weight_power_start=0.05
weight_power_end=2.0
wanda_power_start=0.05
wanda_power_end=2.0
step=0.05

# Define layer names
layer_names=(
    "self_attn.q_proj"
    "self_attn.k_proj"
    "self_attn.v_proj"
    "self_attn.o_proj"
    "mlp.gate_proj"
    "mlp.up_proj"
    "mlp.down_proj"
)

# Create a directory for logs if it doesn't exist
mkdir -p compare_logs

# Log file path
log_file="icml_logs/wanda_new6_layers_llama2_icml.log"

# Write header to the log file
echo "Starting grid test for wanda_power, weight_power, and layer_name" > $log_file
echo "Log format: weight_power=VALUE, wanda_power=VALUE, layer_name=VALUE" >> $log_file
echo "===============================================" >> $log_file

# Loop over weight_power, wanda_power, and layer_name
for layer_name in "${layer_names[@]}"; do
    for weight_power in $(seq $weight_power_start $step $weight_power_end); do
        # Log the current combination
        echo "Running test: weight_power=$weight_power, wanda_power=$wanda_power_start, layer_name=$layer_name" | tee -a $log_file

        # Run the Python script with the specified parameters
        python main.py \
            --model meta-llama/Llama-2-7b-hf \
            --prune_method wanda_new \
            --mode 6 \
            --sparsity_ratio 0.5 \
            --sparsity_type unstructured \
            --weight_power $weight_power \
            --wanda_power $wanda_power_start \
            --layer_name "$layer_name" \
            >> $log_file 2>&1

        # Log separator after each test
        echo "-----------------------------------------------" >> $log_file
        
    done
    for wanda_power in $(seq $wanda_power_start $step $wanda_power_end); do
        # Log the current combination
        echo "Running test: weight_power=$weight_power_start, wanda_power=$wanda_power, layer_name=$layer_name" | tee -a $log_file

        # Run the Python script with the specified parameters
        python main.py \
            --model meta-llama/Llama-2-7b-hf \
            --prune_method wanda_new \
            --mode 6 \
            --sparsity_ratio 0.5 \
            --sparsity_type unstructured \
            --weight_power $weight_power_start \
            --wanda_power $wanda_power \
            --layer_name "$layer_name" \
            >> $log_file 2>&1

        # Log separator after each test
        echo "-----------------------------------------------" >> $log_file
    done
done

echo "Grid testing completed. Logs saved to $log_file"