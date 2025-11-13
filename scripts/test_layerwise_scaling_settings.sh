#!/bin/bash

# Example usage:
# nohup bash test_layerwise_scaling_settings.sh > test_layerwise_scaling_settings.log 2>&1 &

# =============================
#     MODELS TO TEST
# =============================
models=(
    # "decapoda-research/llama-7b-hf"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-70B"
)

# =============================
#   SPARSITY CONFIGURATION
# =============================
sparsity_ratios=(0.5)
# sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

sparsity_types=("unstructured" "2:4" "4:8")
# sparsity_types=("unstructured")

# =============================
#   SEQUENCE LENGTHS TO TEST
# =============================
# seqlen_values=(2048)
seqlen_values=(2048 4096)

# =============================
#   PRUNING METHOD
# =============================
prune_method="wanda"

# =============================
#   LAYERWISE SCALING JSON FILES
# =============================
# Provide paths to JSON files containing layerwise scaling settings
# Paths can be relative to the scripts directory or absolute paths
layerwise_json_files=(
    # "layerwise_scaling_settings/org.json"
    "layerwise_scaling_settings/2025_11_10_1.json"
    # Add more JSON file paths here as needed
)

# =============================
#   OTHER SETTINGS
# =============================
# Whether to save model weights
save_model=false   # Set to false if you don't want to save models

# =============================
#   FUNCTION TO RUN PRUNING
# =============================
run_python_command() {
    local model=$1
    local prune_method=$2
    local sparsity_ratio=$3
    local sparsity_type=$4
    local seqlen=$5
    local json_file=$6

    local model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]')
    
    # Extract settings name from JSON file path
    local settings_name=$(basename "$json_file" .json)
    # Replace any path separators or special chars with underscores
    settings_name=$(echo "$settings_name" | tr '/' '_' | tr '.' '_')

    # Create output directory with settings file name
    local out_dir="../out/${model_name}/${sparsity_type}/${prune_method}/${sparsity_ratio}/layerwise_scaling_${settings_name}/seqlen_${seqlen}/"
    local save_model_dir="../saved_models/${model_name}/${sparsity_type}/${prune_method}/${sparsity_ratio}/layerwise_scaling_${settings_name}/seqlen_${seqlen}/"

    echo "--------------------------------------------"
    echo "Running pruning for:"
    echo " Model: $model"
    echo " Method: $prune_method"
    echo " Sparsity Ratio: $sparsity_ratio"
    echo " Sparsity Type: $sparsity_type"
    echo " Sequence Length: $seqlen"
    echo " Layerwise Scaling JSON: $json_file"
    echo " Settings Name: $settings_name"
    echo " Save Model: $save_model"
    echo "--------------------------------------------"

    # Resolve JSON file path (handle relative paths from scripts directory)
    local json_path="$json_file"
    if [ ! -f "$json_path" ] && [ -f "../$json_file" ]; then
        json_path="../$json_file"
    fi

    # Check if JSON file exists
    if [ ! -f "$json_path" ]; then
        echo "Error: JSON file not found: $json_file"
        echo "Skipping this configuration..."
        echo
        return 1
    fi

    # Build the command dynamically
    cmd="python ../main.py \
        --model \"$model\" \
        --prune_method \"$prune_method\" \
        --sparsity_ratio \"$sparsity_ratio\" \
        --sparsity_type \"$sparsity_type\" \
        --seqlen $seqlen \
        --layerwise_scaling \
        --layerwise_powers_json \"$json_path\" \
        --save \"$out_dir\""

    if [ "$save_model" = true ]; then
        cmd+=" --save_model \"$save_model_dir\""
    fi

    echo "Running command:"
    echo "$cmd"
    echo

    # Run it
    eval $cmd

    echo "Finished: $model | $prune_method | $sparsity_ratio | $sparsity_type | seqlen=$seqlen | settings=$settings_name | save_model=$save_model"
    echo
}

# =============================
#   VALIDATE JSON FILES
# =============================
if [ ${#layerwise_json_files[@]} -eq 0 ]; then
    echo "Error: No JSON files specified in layerwise_json_files array!"
    echo "Please add JSON file paths to the layerwise_json_files array at the top of this script."
    exit 1
fi

echo "Testing with ${#layerwise_json_files[@]} layerwise scaling JSON file(s):"
for json_file in "${layerwise_json_files[@]}"; do
    echo "  - $json_file"
done
echo

# =============================
#   MAIN LOOP
# =============================
for model in "${models[@]}"; do
    for sparsity_ratio in "${sparsity_ratios[@]}"; do
        for sparsity_type in "${sparsity_types[@]}"; do
            for seqlen in "${seqlen_values[@]}"; do
                for json_file in "${layerwise_json_files[@]}"; do
                    run_python_command "$model" "$prune_method" "$sparsity_ratio" "$sparsity_type" "$seqlen" "$json_file"
                done
            done
        done
    done
done

echo "============================================"
echo "All layerwise scaling settings tests completed!"
echo "============================================"
