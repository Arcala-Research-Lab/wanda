CUDA_VISIBLE_DEVICES=0 python wanda/lora_ft/finetune_lm.py \
    --model_name_or_path "/srv/disk00/oyahia/out/wanda_test/wanda0.5" \
    --config_name "baffo32/decapoda-research-llama-7B-hf" \
    --dataset_name mmlu \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir "/srv/disk00/oyahia/out/wanda_test/lora0.5"

# CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
#     --model "/srv/disk00/oyahia/out/wanda_test/wanda0.5" \
#     --lora_weights "/srv/disk00/oyahia/out/wanda_test/lora0.5"