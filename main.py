import argparse
import os 
import numpy as np
import torch
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from contextlib import redirect_stdout

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, prune_mag_mask, prune_wanda_mask
from lib.eval import eval_ppl, eval_zero_shot
from lib.awq_mask import awq_mask
from awq.quantize.pre_quant import run_awq


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
    parser.add_argument('--sparsity_ratios', type=float, nargs='+', default=[], help='Sparsity levels')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[], help='AWQ Mask Thresholds')
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

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
        # unused
        # wanda_mag_intersect_list = []
        # wanda_mag_difference_list = []
        # wanda_awq_intersect_list = []
        # wanda_awq_difference_list = []

        # get lists for all sparsities and thresholds
        mag_lists = []
        awq_lists = []
        # # get all wanda masks for every sparsity ratio
        for sparsity in args.sparsity_ratios:
            W_mag_list = prune_mag_mask(sparsity, model.to(device), tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            W_wanda_list = prune_wanda_mask(sparsity, args, model.to(device), tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            file_name = f'{sparsity}_mag.pt'
            torch.save((W_mag_list, W_wanda_list, sparsity), file_name)
            mag_lists.append(file_name)
            torch.cuda.empty_cache()
        W_mag_list, W_wanda_list, sparsity = torch.load(mag_lists[0])

        # get quantization config (apart from w_bit)
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": 128,  # whether to use group quantization
        }
        w_bit = 4
        ret = run_awq(model, tokenizer, w_bit, q_config)

        # # get all awq masks for every thresholds
        for threshold in args.thresholds:
            W_awq_list = awq_mask(ret['scale'], W_mag_list, threshold)
            file_name = f'{threshold}_awq.pt'
            awq_lists.append(file_name)
            torch.save((W_awq_list, threshold), file_name)

        # converts pre-calculated llama thresholds to their corresponding percentile
        threshold_to_percents = {3.46484375: '1', 2.21875 : '5', 1.8837890625: '10'}

        # gets every combination of magnitude and threshold
        for (mag_file, awq_file) in product(mag_lists, awq_lists):
            W_mag_list, W_wanda_list, sparsity = torch.load(mag_file)
            W_awq_list, threshold = torch.load(awq_file)
            print(f'comparing {threshold}, {sparsity}')
            with open(f'awq_{threshold_to_percents[threshold]}_wanda_{sparsity}', 'w') as log_file:
                with redirect_stdout(log_file):
                    # running totals for the current combination
                    total_mag = 0
                    total_awq = 0
                    total_sum = 0
                    wanda_mag_intersection_sum = 0
                    wanda_awq_intersection_sum = 0

                    for idx, elem in enumerate(W_wanda_list):
                        wanda_flat = elem.flatten()
                        mag_flat = W_mag_list[idx].flatten()
                        awq_flat = W_awq_list[idx].flatten()

                        total_size = wanda_flat.numel()
                        total_sum += total_size

                        # get # of weights that are kept by both wanda and mag/awq
                        wanda_mag_intersection = torch.where((wanda_flat == 0) & (mag_flat == 0), torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
                        wanda_mag_count = wanda_mag_intersection.sum().item()
                        wanda_mag_intersection_sum += wanda_mag_count

                        wanda_awq_intersection = torch.where((wanda_flat == 0) & (awq_flat == 0), torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
                        wanda_awq_count = wanda_awq_intersection.sum().item()
                        wanda_awq_intersection_sum += wanda_awq_count

                        # get # of weights that are kept by mag/awq
                        mag_zeroes = torch.where(mag_flat == 0, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
                        num_wanda_mag = mag_zeroes.sum().item()
                        total_mag += num_wanda_mag

                        awq_zeroes = torch.where(awq_flat == 0, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
                        num_wanda_awq = awq_zeroes.sum().item()
                        total_awq += num_wanda_awq

                        # Elements in tensor1 but not in tensor2 (unused)
                        # wanda_mag_difference = total_size - wanda_mag_count
                        # wanda_awq_difference = total_size - wanda_awq_count

                        # the percent corresponds to the following question:
                        # out of all the weights awq chose as salient, how many did wanda also determine as salient?
                        wanda_awq_percent = wanda_awq_count*100/num_wanda_awq if num_wanda_awq != 0 else 'N/A (all below threshold)'

                        sparsity_size = total_size*sparsity
                        print(f"layer {idx}: total_size: {total_size} sparsity_size: {sparsity_size} AWQ threshold: {threshold_to_percents[threshold]}%")
                        print(f"(wanda, mag) intersection: {wanda_mag_count} %:{wanda_mag_count*100/num_wanda_mag}")
                        print(f"(wanda, awq) intersection: {wanda_awq_count} %:{wanda_awq_percent}")

                    print("---------------")
                    assert wanda_mag_intersection_sum <= total_mag, "Numerator exceeds denominator!"
                    sparsity_size = total_sum*sparsity
                    print(f"Total magnitude kept weights: {total_mag} awq kept weights: {total_awq}")
                    print(f"Total total_size: {total_sum} sparsity_size: {sparsity_size} AWQ threshold: {threshold_to_percents[threshold]}%")
                    print(f"(wanda, mag) intersection: {wanda_mag_intersection_sum} %:{wanda_mag_intersection_sum*100/total_mag}")
                    print(f"(wanda, awq) intersection: {wanda_awq_intersection_sum} %:{wanda_awq_intersection_sum*100/total_awq}")
                    torch.cuda.empty_cache()
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