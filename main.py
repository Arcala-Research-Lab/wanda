import argparse
import os 
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, prune_mag_mask, prune_wanda_mask
from lib.eval import eval_ppl, eval_zero_shot
from lib.awq_mask import awq_mask
from lib.awq_pre_quant_no_apply import run_awq

from awq.quantize.quantizer import real_quantize_model_weight
from awq.utils.utils import simple_dispatch_model
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_in_model,
)
from datasets import load_dataset

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--eval_seqlen", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    
    # AWQ args
    parser.add_argument("--load_quant", default=None, type=str, help='Path to previously computed AWQ quantization results')
    parser.add_argument("--w_bit", type=int, default=None)
    parser.add_argument("--q_group_size", type=int, default=-1)
    parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")

    parser.add_argument("--eval_zero_shot", action="store_true")

    parser.add_argument("--compare_selection", action="store_true")

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    """pasted from llm-awq entry.py"""
    if args.load_quant:  # directly load quantized weights
        q_config = {
            "zero_point": not args.no_zero_point,  # by default True
            "q_group_size": args.q_group_size,  # whether to use group quantization
        }

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, use_fast=False, trust_remote_code=True
            )

        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)
        model.seqlen = model.config.max_position_embeddings 

        model.eval()
    else:
        model_name = args.model.split("/")[-1]
        print(f"loading llm model {args.model}")
        model = get_llm(args.model, args.cache_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # ARCALA: Compare weight selection
    if args.compare_selection:
        wanda_mag_intersect_list = []
        wanda_mag_difference_list = []
        wanda_awq_intersect_list = []
        wanda_awq_difference_list = []

        total_sum = 0
        wanda_mag_intersection_sum = 0
        wanda_awq_intersection_sum = 0

        # get quantization config (apart from w_bit)
        # TODO: put awq_enabled, w_bit, group_size and no_zero_point as config params
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": 128,  # whether to use group quantization
        }
        w_bit = 4
        ret = run_awq(model, tokenizer, w_bit, q_config)

        # AWQ uses a different device than wanda, so I did .to(device) on everything was consistent
        W_mag_list = prune_mag_mask(args, model.to(device), tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        W_wanda_list = prune_wanda_mask(args, model.to(device), tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        W_awq_list = awq_mask(ret['scale'], W_mag_list)
        for idx, elem in enumerate(W_wanda_list):
            wanda_flat = elem.flatten()
            mag_flat = W_mag_list[idx].flatten()
            awq_flat = W_awq_list[idx].flatten()

            total_size = wanda_flat.numel()
            total_sum += total_size

            wanda_mag_intersection = torch.where(wanda_flat == mag_flat, wanda_flat, torch.tensor(0, device='cuda'))
            wanda_mag_count = wanda_mag_intersection.sum().item()
            wanda_mag_intersection_sum += wanda_mag_count

            wanda_awq_intersection = torch.where(wanda_flat == awq_flat, wanda_flat, torch.tensor(0, device='cuda'))
            wanda_awq_count = wanda_awq_intersection.sum().item()
            wanda_awq_intersection_sum += wanda_awq_count

            # Elements in tensor1 but not in tensor2
            wanda_mag_difference = total_size - wanda_mag_count
            wanda_awq_difference = total_size - wanda_awq_count

            sparsity_size = total_size *args.sparsity_ratio

            print(f"layer {idx}: total_size: {total_size} sparsity_size: {sparsity_size}")
            print(f"(wanda, mag) intersection: {wanda_mag_count} %:{wanda_mag_count*100/sparsity_size}")
            print(f"(wanda, awq) intersection: {wanda_awq_count} %:{wanda_awq_count*100/sparsity_size}")

        print("---------------")
        sparsity_size = total_sum*args.sparsity_ratio
        print(f"Total total_size: {total_sum} sparsity_size: {sparsity_size}")
        print(f"(wanda, mag) intersection: {wanda_mag_intersection_sum} %:{wanda_mag_intersection_sum*100/sparsity_size}")
        print(f"(wanda, awq) intersection: {wanda_awq_intersection_sum} %:{wanda_awq_intersection_sum*100/sparsity_size}")
        exit()
        pass

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    model.seqlen = args.eval_seqlen if args.eval_seqlen else model.seqlen # for evaluating perplexity with specific seqlen
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()