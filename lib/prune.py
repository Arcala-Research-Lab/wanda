import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 


#####################################
import numpy as np
import matplotlib.pyplot as plt
import os


def save_activation_distribution(activations, title, filename, bins=100):
    """
    Save histogram of activation distribution.
    """
    plt.figure(figsize=(10, 6))
    
    acts = activations.cpu().numpy().flatten()
    
    # Compute statistics
    mean_val = acts.mean()
    std_val = acts.std()
    threshold_pos = mean_val + 3 * std_val
    threshold_neg = mean_val - 3 * std_val
    
    plt.hist(acts, bins=bins, alpha=0.7, color='#3498DB', edgecolor='black')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(threshold_pos, color='orange', linestyle='--', linewidth=2, label=f'+3σ: {threshold_pos:.4f}')
    plt.axvline(threshold_neg, color='orange', linestyle='--', linewidth=2, label=f'-3σ: {threshold_neg:.4f}')
    
    plt.xlabel('Raw Activation Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def compute_activation_stats(activations):
    """Compute statistics for activations."""
    acts = activations.flatten()
    mean_val = acts.mean().item()
    std_val = acts.std().item()
    kurtosis = (((acts - mean_val) / std_val) ** 4).mean().item()
    max_mean_ratio = (acts.max() / mean_val).item()
    pct_outliers = 100 * (acts > mean_val + 3*std_val).float().mean().item()
    
    return {
        'mean': mean_val,
        'std': std_val,
        'kurtosis': kurtosis,
        'max_mean_ratio': max_mean_ratio,
        'pct_outliers': pct_outliers
    }

def extract_raw_activations(wrapped_layers):
    """Extract raw activations from first 2 batches."""
    raw_act_dict = {}
    
    for name, wrapper in wrapped_layers.items():
        if len(wrapper.raw_activations) > 0:
            # Concatenate first 2 batches and flatten
            acts = torch.cat(wrapper.raw_activations, dim=0)  # (2, seq_len, in_features)
            acts_flat = acts.reshape(-1, acts.shape[-1])  # (2*seq_len, in_features)
            raw_act_dict[name] = acts_flat
    
    return raw_act_dict


def analyze_layer_distributions(i, name, weights, W_mask, outlier_mask_full, filename_prefix, output_dir='distributions'):
    """Analyze and save distributions for a single layer - REMOVED PER-LAYER PLOTS."""
    # Just return data for aggregation, no per-layer plots
    return weights.flatten().cpu(), W_mask.flatten().cpu(), outlier_mask_full.flatten().cpu()


def compute_outlier_mask(activations, weights_shape, threshold_sigma=3.0):
    """
    Compute which weights correspond to outlier activations (>mean + 3*std).
    
    Args:
        activations: scaler_row tensor, shape (in_features,) or (1, in_features)
        weights_shape: shape of weight tensor (out_features, in_features)
        threshold_sigma: number of std devs above mean to be considered outlier
    
    Returns:
        Boolean mask of shape (out_features, in_features) where True = outlier activation
    """
    acts = activations.flatten()
    mean_act = acts.mean()
    std_act = acts.std()
    threshold = mean_act + threshold_sigma * std_act
    
    # Create mask: True where activation is outlier
    outlier_act_mask = acts > threshold
    
    # Broadcast to weight shape (each column gets same outlier status)
    outlier_mask = outlier_act_mask.unsqueeze(0).expand(weights_shape[0], -1)
    
    return outlier_mask
#####################################

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

# ARCALA function to create magnitude mask
def prune_mag_mask(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    W_mag_mask_list = []

    # magnitude based pruning
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W_mag_mask_list.append(W_mask)

    return W_mag_mask_list



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    # trying wikitext loader
    # dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    mask_index = 0
    scales_index = 0
    s = []


    ##################################### Initialize structures
    # type_weights = {}
    # type_masks = {} 
    # type_outlier_masks = {} 
    activation_stats = {} 
    
    if args.layerwise_scaling:
        if args.sparsity_type != "unstructured":
            filename_prefix = f"structured_{prune_n}_{prune_m}_layerwise"
            print(f"--- Running LAYERWISE SCALING with STRUCTURED {prune_n}:{prune_m} mode ---")
        else:
            filename_prefix = "unstructured_layerwise" 
            print("--- Running LAYERWISE SCALING with UNSTRUCTURED mode ---")
    else:
        if args.sparsity_type != "unstructured":
            filename_prefix = f"structured_{prune_n}_{prune_m}_wanda"
            print(f"--- Running in STRUCTURED {prune_n}:{prune_m} WANDA mode ---")
        else:
            filename_prefix = "normal_wanda"
            print("--- Running in NORMAL WANDA mode (layerwise_scaling=False) ---")
    #####################################

    if args.awq_mask:
        awq_mask = torch.load(args.awq_mask)
    if args.awq_scales:
        awq_scales = torch.load(args.awq_scales)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
            if args.awq_scales:
                if args.normalize:
                    weight = torch.abs(subset[name].weight.data)
                    awq_indiv = awq_scales[scales_index].to(weight.device)
                    wanda_scales = wrapped_layers[name].scaler_row
                    # scale wanda by AWQ min-max
                    wanda_scales = (wanda_scales-wanda_scales.min())/(wanda_scales.max()-wanda_scales.min())\
                        *(awq_indiv.max()-awq_indiv.min())+awq_indiv.min()
                    W_metric = weight * awq_indiv.reshape(1, -1) * wanda_scales.reshape(1, -1)
                else:
                    if args.scale_and_wmetric:
                        W_metric = torch.abs(subset[name].weight.data) * awq_scales[scales_index].reshape((1, -1)).to(subset[name].weight.data.device) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                    elif args.wmetric_and_scale:
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) * awq_scales[scales_index].reshape((1, -1)).to(subset[name].weight.data.device)
                    else:
                        W_metric = torch.abs(subset[name].weight.data) * awq_scales[scales_index].reshape((1, -1)).to(subset[name].weight.data.device)
                scales_index += 1
            elif args.layerwise_scaling:
                weights = subset[name].weight.data  # <-- Get the raw weights (with negative values)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

                # W_metric is calculated based on the *absolute* value
                if name == 'mlp.gate_proj' or name == 'mlp.up_proj':
                    W_metric = torch.pow(torch.abs(weights), 1)  * torch.pow(wanda_scale, 0.1)
                elif name == 'self_attn.v_proj':
                    W_metric = torch.pow(torch.abs(weights), 1.75)  * torch.pow(wanda_scale, 1)
                elif name == 'self_attn.o_proj':
                    W_metric = torch.pow(torch.abs(weights), 1.25)  * torch.pow(wanda_scale, 1)
                elif name == 'mlp.down_proj':
                    W_metric = torch.pow(torch.abs(weights), 1.75)  * torch.pow(wanda_scale, 1)
                else:
                    W_metric = torch.abs(weights) * wanda_scale
            else:

                if args.capture_scaler_row:
                    s.append(wrapped_layers[name].scaler_row)
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))               

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            if args.awq_mask:
                W_mask = torch.logical_and(W_mask, torch.logical_not(awq_mask[mask_index]).reshape(W_mask.shape[0], W_mask.shape[1]).to(W_mask.device))
                mask_index += 1

            #################################### Computing step for activation statistics
            # weights = subset[name].weight.data  
            
            # Compute outlier mask based on activations
            activations = wrapped_layers[name].scaler_row
            # outlier_mask = compute_outlier_mask(activations, weights.shape, threshold_sigma=3.0)

            # Collect activation statistics
            if name not in activation_stats:
                activation_stats[name] = {
                    'raw_activations': [], 
                    'mean': [], 'std': [], 'kurtosis': [], 
                    'max_mean_ratio': [], 'pct_outliers': []
                }
            
            if len(wrapped_layers[name].raw_activations) > 0:
                acts = torch.cat(wrapped_layers[name].raw_activations, dim=0)  # (2, 2048, 4096)
                acts_flat = acts.reshape(-1, acts.shape[-1]).flatten()  # Flatten all values
                activation_stats[name]['raw_activations'].append(acts_flat.cpu())
                
                # Compute stats on raw activations (use float64 to avoid overflow)
                acts_flat_f64 = acts_flat.double()  # Convert to float64
                mean_val = acts_flat_f64.mean().item()
                std_val = acts_flat_f64.std().item()
                kurtosis = (((acts_flat_f64 - mean_val) / (std_val + 1e-10)) ** 4).mean().item()
                max_mean_ratio = (acts_flat_f64.abs().max() / (abs(mean_val) + 1e-10)).item()
                threshold = mean_val + 3 * std_val
                pct_outliers = 100 * (acts_flat_f64 > threshold).float().mean().item()
                
                activation_stats[name]['mean'].append(mean_val)
                activation_stats[name]['std'].append(std_val)
                activation_stats[name]['kurtosis'].append(kurtosis)
                activation_stats[name]['max_mean_ratio'].append(max_mean_ratio)
                activation_stats[name]['pct_outliers'].append(pct_outliers)

            # w_samples, mask_samples, outlier_samples = analyze_layer_distributions(
            #     i, name, weights, W_mask, outlier_mask, filename_prefix
            # )
            
            # if name not in type_masks:
            #     type_weights[name] = []
            #     type_masks[name] = []
            #     type_outlier_masks[name] = []
            # type_weights[name].append(w_samples)
            # type_masks[name].append(mask_samples)
            # type_outlier_masks[name].append(outlier_samples)

            del W_metric
            if args.layerwise_scaling: 
                del wanda_scale
            torch.cuda.empty_cache()    
            ####################################
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    if args.capture_scaler_row:
        torch.save(s, 'out/wanda_scales')
    ##################################### Saving activation statistics summary
    print("\n" + "="*80)
    print(f"ACTIVATION STATISTICS SUMMARY - {filename_prefix}")
    print("="*80)
    print(f"{'Layer Type':<20} {'Kurtosis':<12} {'Max/Mean':<12} {'%Outliers':<12} {'Mean':<12} {'Std':<12}")
    print("-"*80)
    
    if activation_stats:
        os.makedirs('distributions/activations', exist_ok=True)
        
        for layer_name in sorted(activation_stats.keys()):
            stats = activation_stats[layer_name]
            
            # Compute averages
            avg_kurtosis = np.mean(stats['kurtosis'])
            avg_max_mean = np.mean(stats['max_mean_ratio'])
            avg_pct_outliers = np.mean(stats['pct_outliers'])
            avg_mean = np.mean(stats['mean'])
            avg_std = np.mean(stats['std'])
            
            print(f"{layer_name:<20} {avg_kurtosis:<12.2f} {avg_max_mean:<12.2f} {avg_pct_outliers:<12.2f} {avg_mean:<12.6f} {avg_std:<12.6f}")
            
            if len(stats.get('raw_activations', [])) > 0:
                all_activations = torch.cat(stats['raw_activations'])
                safe_name = layer_name.replace('.', '_')
                
                act_filename = f'distributions/activations/{filename_prefix}_{safe_name}_activation_distribution.png'
                act_title = f'{layer_name} - {filename_prefix} - Raw Activation Distribution'
                save_activation_distribution(all_activations, act_title, act_filename)
    
    print("="*80)
    if activation_stats:
        os.makedirs('stats', exist_ok=True)
        csv_filename = f'stats/{filename_prefix}_activation_dist.csv'
        
        import csv
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer Type', 'Mean', 'Std', 'Kurtosis', 'Max/Mean', '%Outliers'])
            
            for layer_name in sorted(activation_stats.keys()):
                stats = activation_stats[layer_name]
                avg_mean = np.mean(stats['mean'])
                avg_std = np.mean(stats['std'])
                avg_kurtosis = np.mean(stats['kurtosis'])
                avg_max_mean = np.mean(stats['max_mean_ratio'])
                avg_pct_outliers = np.mean(stats['pct_outliers'])
                
                writer.writerow([layer_name, avg_mean, avg_std, avg_kurtosis, avg_max_mean, avg_pct_outliers])
        
        print(f"Activation statistics saved to: {csv_filename}\n")
    ######################################
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

# ARCALA function to extract pruning mask

def prune_wanda_mask(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    W_wanda_mask_list = []

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    # trying wikitext loader
    # dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            W_wanda_mask_list.append(W_mask)
            # subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        # inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    return W_wanda_mask_list


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()