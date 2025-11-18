#!/bin/bash

# --- Configuration ---
# Base directory for plots (relative to scripts/)
BaseDir="distributions/activations"
CompareDir="comparisons_activations"
MasterFile="MASTER_ACTIVATION_COMPARISON.png"

# Prefixes for the two methods
NormalPrefix="normal_wanda"
LayerwisePrefix="unstructured_layerwise"

# List of all layer names
LayerNames=(
    "mlp_gate_proj"
    "mlp_down_proj"
    "mlp_up_proj"
    "self_attn_q_proj"
    "self_attn_k_proj"
    "self_attn_v_proj"
    "self_attn_o_proj"
)

# Common suffix for activation distribution files
FileSuffix="_activation_distribution.png"

# --- 1. Create the comparisons directory if it doesn't exist ---
if [ ! -d "$CompareDir" ]; then
    mkdir -p "$CompareDir"
    echo "Created '$CompareDir' directory."
fi

# --- 2. Create the 7 paired images ---
echo "--- Step 1: Creating 7 paired activation comparison images ---"
PairedFiles=()

for layer in "${LayerNames[@]}"; do
    # Construct the full paths
    NormalFile="${BaseDir}/${NormalPrefix}_${layer}${FileSuffix}"
    LayerwiseFile="${BaseDir}/${LayerwisePrefix}_${layer}${FileSuffix}"
    OutFile="${CompareDir}/compare_activations_${layer}.png"
    
    # Check if both input files exist
    if [ -f "$NormalFile" ] && [ -f "$LayerwiseFile" ]; then
        echo "Montaging: $layer"
        # Create side-by-side comparison with labels
        montage \
            -label "Normal WANDA" "$NormalFile" \
            -label "Unstructured Layerwise" "$LayerwiseFile" \
            -tile 2x1 -geometry +5+5 \
            -pointsize 14 \
            "$OutFile"
        PairedFiles+=("$OutFile")
    else
        echo -e "\033[33mWarning: Skipping '$layer'. Missing input file(s).\033[0m"
        [ ! -f "$NormalFile" ] && echo "  Missing: $NormalFile"
        [ ! -f "$LayerwiseFile" ] && echo "  Missing: $LayerwiseFile"
    fi
done

# --- 3. Create the final master grid ---
echo "--- Step 2: Creating master activation dashboard image ---"
if [ ${#PairedFiles[@]} -eq 0 ]; then
    echo -e "\033[31mError: No paired files were created. Cannot build master image.\033[0m"
    echo "Please check that your '$BaseDir' directory contains the correct files."
else
    # We use 2x4 to fit all 7 images cleanly in a grid
    montage "${PairedFiles[@]}" \
        -tile 2x4 \
        -geometry +10+10 \
        -title "Normal WANDA vs Unstructured Layerwise - Activation Distributions" \
        -pointsize 20 \
        "$MasterFile"
    
    echo -e "\033[32m----------------------------------------"
    echo "Success! Your master comparison image is ready:"
    echo "$MasterFile"
    echo -e "----------------------------------------\033[0m"
    echo -e "\033[36mIndividual comparisons saved in: $CompareDir/\033[0m"
fi