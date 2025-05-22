#!/bin/bash

# Set common variables
# model="decapoda-research/llama-7b-hf"
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5
cuda_device=1

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    /home/oyahia/.conda/envs/RLPruner/bin/python wanda/main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio_weights $2 \
    --sparsity_ratio_activations $3 \
    --sparsity_type $4 \
    --save $5 \
    --save_model $6 \
    --eval_seqlen 4096 \
    --nsamples 32 \
    > $7
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
# > "/srv/disk00/oyahia/out/wanda_test/wanda0.6ppl"
# run_python_command "salient" "0.5" "0.5" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.5_salient.txt"
# run_python_command "salient" "0.3" "0.7" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.3_salient.txt"
# run_python_command "salient" "0.4" "0.6" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.4_salient.txt"
# run_python_command "salient" "0.45" "0.55" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.45_salient.txt"
# run_python_command "salient" "0.46" "0.54" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.46_salient.txt"
# run_python_command "salient" "0.47" "0.53" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.47_salient.txt"
# run_python_command "bad_wanda" "0.99" "0.99" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.01_bad_wanda.txt"
# run_python_command "bad_magnitude" "0.99" "0.99" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.01_bad_mag.txt"
# run_python_command "bad_wanda" "0.9762" "0.9762" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.02_bad_wanda.txt"
# run_python_command "bad_magnitude" "0.9762" "0.9762" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.02_bad_mag.txt"
run_python_command "fix_mag" "0.65" "0.05" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.65_fix_mag_0.05.txt"
# run_python_command "magnitude" "0.5" "0.5" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.5_magnitude.txt"
# run_python_command "salient" "0.6" "0.4" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.6_salient.txt"
# run_python_command "magnitude" "0.6" "0.4" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.6_magnitude.txt"
# run_python_command "salient" "0.55" "0.45" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.55_salient.txt"
# run_python_command "magnitude" "0.55" "0.45" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.55_magnitude.txt"
# run_python_command "magnitude" "0.51" "0.49" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.51_magnitude.txt"
# run_python_command "bad_wanda" "0.9497" "0.9497" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.05_bad_wanda.txt"
# run_python_command "bad_magnitude" "0.9497" "0.9497" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.05_bad_mag.txt"
# run_python_command "salient" "0.51" "0.49" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.51_salient.txt"
# run_python_command "magnitude" "0.52" "0.48" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.52_magnitude.txt"
# run_python_command "salient" "0.52" "0.48" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.52_salient.txt"
# run_python_command "magnitude" "0.53" "0.47" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.53_magnitude.txt"
# run_python_command "salient" "0.53" "0.47" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.53_salient.txt"
# run_python_command "magnitude" "0.54" "0.46" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.54_magnitude.txt"
# run_python_command "salient" "0.54" "0.46" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.54_salient.txt"
# run_python_command "salient" "0.57" "0.43" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.57_salient.txt"
# run_python_command "magnitude" "0.57" "0.43" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.57_magnitude.txt"
# run_python_command "salient" "0.2" "0.8" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.2_salient.txt"
# run_python_command "magnitude" "0.2" "0.8" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.2_magnitude.txt"
# run_python_command "salient" "0.1" "0.9" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.1_salient.txt"
# run_python_command "magnitude" "0.1" "0.9" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.1_magnitude.txt"
# run_python_command "salient" "0.05" "0.95" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.05_salient.txt"
# run_python_command "magnitude" "0.05" "0.95" "unstructured" "/srv/disk00/oyahia/out/wanda_test/wanda" "/srv/disk00/oyahia/out/wanda_test/wanda" "out/salient_out/0.05_magnitude.txt"
echo "Finished wanda pruning method"

# # llama-7b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
# run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/"
# run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/"
# echo "Finished magnitude pruning method"