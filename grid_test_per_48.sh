#!/bin/bash

# Create a directory for logs if it doesn't exist
mkdir -p compare_logs

# Log file path
log_file="compare_logs/wanda_new17_per_layer_num.log"

# Write header to the log file
echo "Starting grid test for wanda_power, weight_power, and layer_name" > $log_file
echo "Log format: weight_power=VALUE, wanda_power=VALUE, layer_name=VALUE" >> $log_file
echo "===============================================" >> $log_file

# Loop over weight_power, wanda_power, and layer_name
for layer_name in $(seq 0 31); do
    # Log the current combination
    echo "Running test: layer_name=$layer_name" | tee -a $log_file

    # Run the Python script with the specified parameters
    python main.py \
        --model meta-llama/Llama-2-7b-hf \
        --prune_method wanda_new \
        --mode 17 \
        --sparsity_ratio 0.5 \
        --sparsity_type 4:8 \
        --layer_name "$layer_name" \
        >> $log_file 2>&1

    # Log separator after each test
    echo "-----------------------------------------------" >> $log_file
done

echo "Grid testing completed. Logs saved to $log_file"