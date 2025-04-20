import sys
sys.path.append("../")

from lib.eval import eval_ppl
from peft import PeftModel, PeftConfig, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

#Evaluate perplexity of Pruned models (works for LoRA models too, but LoRA model used must not be merged - i.e. you haven't used save_lora.py on the model)

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16, cache_dir=args.cache_dir, low_cpu_mem_usage=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, token=HF_TOKEN)
    print(f"SEQLEN: {args.ctx_length}")
    if args.lora_weights:
        model = PeftModel.from_pretrained(model,args.lora_weights,torch_dtype=torch.float16)
        set_peft_model_state_dict(model, torch.load(os.path.join(args.lora_weights, 'adapter_model.bin')))
        print("fine-tuned ", end="")

    model.eval()
    model.seqlen = args.ctx_length
    ppl = eval_ppl(args, model, tokenizer)
    print(f"perplexity on wikitext {ppl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str
    )
    parser.add_argument(
        '--cache_dir', type=str, default="llm_weights"  
    )
    parser.add_argument(
        '--lora_weights', type=str, default=None 
    )
    parser.add_argument(
        '--ctx_length', type=int, default=4096 
    )
    parser.add_argument("--eval_zero_shot", action="store_true")

    args = parser.parse_args()
    main(args)