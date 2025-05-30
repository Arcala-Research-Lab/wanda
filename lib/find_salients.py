# pylint: disable=consider-using-enumerate
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .prune import prepare_calibration_input, find_layers
import torch
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def set_submodule(module, path, new_module):
    """
    Replaces a nested submodule in a model.
    For example, path = "attn.q_proj" will replace module.attn.q_proj
    """
    parts = path.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    setattr(module, parts[-1], new_module)

def find_salients(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    w_sparsity = args.sparsity_ratio_weights
    salient_sparsity = args.sparsity_ratio_activations

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

    OUT_PATH = Path(f'/home/oyahia/yeh-research/out/raw_percents')
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric_w = torch.abs(subset[name].weight.data)
            W_metric_a = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row)

            def get_wmask(W_metric: float, sparsity_ratio: float):
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                return W_mask

            W_mask_a = get_wmask(W_metric_a, salient_sparsity)
            min_weights = min(w_sparsity, 1-salient_sparsity)

            W_mask_w = get_wmask(W_metric_w, w_sparsity)
            W_mask_salient = W_mask_w & ~W_mask_a

            if args.prune_method == "salient":
                subset[name].weight.data[W_mask_salient] = 0
            elif args.prune_method == "magnitude":
                subset[name].weight.data[W_mask_w] = 0
            elif args.prune_method == "wanda":
                subset[name].weight.data[W_mask_a] = 0
            elif args.prune_method == "bad_magnitude":
                subset[name].weight.data[~W_mask_w] = 0
            elif args.prune_method == "bad_wanda":
                subset[name].weight.data[~W_mask_a] = 0
            elif args.prune_method == "salient_random":
                readjusted_mask = W_mask_w.clone()
                one_indices = torch.nonzero(readjusted_mask, as_tuple=True)
                zero_indices = torch.nonzero(~readjusted_mask, as_tuple=True)
                num_ones = one_indices[0].numel()
                num_zeros = zero_indices[0].numel()
                num_ones_to_move = int(num_ones * 0.1)
                num_ones_to_move = min(num_ones_to_move, num_ones, num_zeros)
                perm_ones = torch.randperm(num_ones)
                indices_to_flip_to_false_in_original_ones = tuple(idx[perm_ones[:num_ones_to_move]] for idx in one_indices)
                perm_zeros = torch.randperm(num_zeros)
                indices_to_flip_to_true_in_original_zeros = tuple(idx[perm_zeros[:num_ones_to_move]] for idx in zero_indices)
                readjusted_mask[indices_to_flip_to_false_in_original_ones] = 0
                readjusted_mask[indices_to_flip_to_true_in_original_zeros] = 1
                subset[name].weight.data[readjusted_mask] = 0
            elif args.prune_method == "fix_mag":
                W_mask_w_0_52 = get_wmask(W_metric_w, 0.52)
                W_mask_a_0_52 = get_wmask(W_metric_a, 0.48)
                W_mask_salient = W_mask_w_0_52 & ~W_mask_a_0_52
                subset[name].weight.data[W_mask_w & ~W_mask_salient] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()