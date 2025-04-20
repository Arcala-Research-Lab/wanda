import sys
sys.path.append("../")

from lib.eval import eval_ppl
from peft import PeftModel, PeftConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

#LoRA fine tuning script keeps the pruned model and the A/B matrices separate
#This script merges them into one model

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16, cache_dir=args.cache_dir, low_cpu_mem_usage=True, device_map="auto")
    if args.lora_weights:
        model = PeftModel.from_pretrained(model, args.lora_weights, torch_dtype=torch.float16)
        set_peft_model_state_dict(model, torch.load(os.path.join(args.lora_weights, 'adapter_model.bin')))
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.save_path)

    else:
        print("LoRA weights not defined")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--cache_dir', type=str, default="llm_weights"  )
    parser.add_argument('--lora_weights', type=str, default=None )
    parser.add_argument('--save_path', type=str, default=None )

    args = parser.parse_args()
    main(args)