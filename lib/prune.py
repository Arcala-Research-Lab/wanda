import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .eval import eval_ppl_wikitext

from .ablate import AblateGPT 

# from awq.quantize.qmodule import WQLinear

def evaluate_perplexity_for_optimization(model, tokenizer, device, max_length=2048):
    """
    Quick perplexity evaluation for optimization purposes.
    Uses a smaller sample size for faster evaluation during grid search.
    """
    from .data import get_loaders
    
    # Get a small test dataset for quick evaluation  
    _, testloader = get_loaders("wikitext2", seed=0, seqlen=max_length, tokenizer=tokenizer)
    
    # Evaluate with smaller batch for speed
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, bs=1, device=device)
    
    return ppl

# Auto-search functions for optimal pruning parameters
@torch.no_grad()
def auto_search_prune_params(layer, subset, wrapped_layers, input_batch, 
                           layer_kwargs, sparsity_ratio, layer_name='all',
                           weight_power_range=(0.5, 2.0), wanda_power_range=(0.5, 2.0),
                           n_grid=20, optimization_metric='mse', model=None, tokenizer=None, 
                           device=None, args=None):
    """
    Automatically search for optimal weight_power and wanda_power parameters
    by minimizing MSE between original and pruned layer outputs or by minimizing perplexity.
    
    Args:
        layer: The transformer layer to optimize
        subset: Dictionary of linear layers in the layer
        wrapped_layers: Dictionary of WrappedGPT objects for activation collection
        input_batch: Input tensor for the layer
        layer_kwargs: Keyword arguments for layer forward pass
        sparsity_ratio: Target sparsity ratio
        layer_name: Specific layer name to optimize ('all' for all layers)
        weight_power_range: (min, max) range for weight_power search
        wanda_power_range: (min, max) range for wanda_power search
        n_grid: Number of grid points to search
        optimization_metric: 'mse' or 'perplexity' - optimization metric to use
        model: Full model (required for perplexity optimization)
        tokenizer: Tokenizer (required for perplexity optimization)  
        device: Device (required for perplexity optimization)
        args: Arguments (required for perplexity optimization)
        
    Returns:
        tuple: (best_weight_power, best_wanda_power, best_error)
    """
    
    # Get original layer output (baseline)
    with torch.no_grad():
        org_out = layer(input_batch, **layer_kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]
    
    # Store original state dict
    org_sd = {k: v.cpu().clone() for k, v in layer.state_dict().items()}
    
    best_error = float("inf")
    best_weight_power = -1
    best_wanda_power = -1
    
    # Generate search grids
    weight_powers = torch.linspace(weight_power_range[0], weight_power_range[1], n_grid)
    wanda_powers = torch.linspace(wanda_power_range[0], wanda_power_range[1], n_grid)
    
    print(f"Searching {n_grid}x{n_grid} grid for optimal pruning parameters...")
    
    for i, weight_power in enumerate(weight_powers):
        for j, wanda_power in enumerate(wanda_powers):
            # Restore original weights
            layer.load_state_dict(org_sd)
            
            # Apply pruning with current parameters temporarily
            original_weights = _apply_prune_with_params_temp(subset, wrapped_layers, sparsity_ratio, 
                                                           weight_power.item(), wanda_power.item(), layer_name)
            
            # Get pruned output
            with torch.no_grad():
                pruned_out = layer(input_batch, **layer_kwargs)
                if isinstance(pruned_out, tuple):
                    pruned_out = pruned_out[0]
            
            # Calculate MSE loss
            loss = (org_out - pruned_out).float().pow(2).mean().item()
            
            # Update best parameters if this is better
            if loss < best_error:
                best_error = loss
                best_weight_power = weight_power.item()
                best_wanda_power = wanda_power.item()
            
            # Restore weights for next iteration
            for name in subset:
                subset[name].weight.data.copy_(original_weights[name])
            
            if (i * n_grid + j + 1) % (n_grid * 2) == 0:  # Progress update
                print(f"Progress: {i * n_grid + j + 1}/{n_grid * n_grid}, "
                      f"Current best: wp={best_weight_power:.3f}, "
                      f"wd={best_wanda_power:.3f}, error={best_error:.6f}")
    
    # Restore original state
    layer.load_state_dict(org_sd)
    print(f"Best parameters found: weight_power={best_weight_power:.3f}, "
          f"wanda_power={best_wanda_power:.3f}, error={best_error:.6f}")
    
    return best_weight_power, best_wanda_power, best_error


def _apply_prune_with_params(subset, wrapped_layers, sparsity_ratio, 
                           weight_power, wanda_power, layer_name):
    """
    Apply pruning to subset layers with given parameters
    """
    for name in subset:
        if layer_name != 'all' and name != layer_name:
            # Use default parameters for non-target layers
            weights = torch.abs(subset[name].weight.data)
            wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = weights * wanda_scale
        else:
            # Use optimized parameters for target layers
            weights = torch.abs(subset[name].weight.data)
            wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
        
        # Create pruning mask
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)
        
        # Apply pruning
        subset[name].weight.data[W_mask] = 0


def _apply_prune_with_params_temp(subset, wrapped_layers, sparsity_ratio, 
                                weight_power, wanda_power, layer_name):
    """
    Temporarily apply pruning to subset layers with given parameters (for testing)
    Returns the modified weights without permanently changing them
    """
    original_weights = {}
    
    for name in subset:
        # Store original weights
        original_weights[name] = subset[name].weight.data.clone()
        
        if layer_name != 'all' and name != layer_name:
            # Use default parameters for non-target layers
            weights = torch.abs(subset[name].weight.data)
            wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = weights * wanda_scale
        else:
            # Use optimized parameters for target layers
            weights = torch.abs(subset[name].weight.data)
            wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
        
        # Create pruning mask
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
        W_mask.scatter_(1, indices, True)
        
        # Apply pruning temporarily
        subset[name].weight.data[W_mask] = 0
    
    return original_weights

# def find_layers(module, layers=[nn.Linear, WQLinear], name=''):
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
        # if isinstance(module, WQLinear):
        #     module.weight = module.qweight
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

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


# def prune_wanda_new(args, model, tokenizer,  device=torch.device("cuda:0"), prune_n=0, prune_m=0, mode=2, weight_power=1, wanda_power=1, awq_power=1, layer_name='all'):
#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     layers = model.model.layers
#     awq_scale = torch.load('/home/tomyeh/ARCALA/wanda/awq_scales')
#     print(f"awq_scale type: {type(awq_scale)}", flush=True)
#     print(f"awq_scale length or shape: {len(awq_scale) if isinstance(awq_scale, list) else awq_scale.shape}", flush=True)

#     # Load calibration data once
#     print("loading calibration data")
#     dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
#     print("dataset loading complete")
#     with torch.no_grad():
#         inps_init, _, attention_mask_init, position_ids_init = prepare_calibration_input(model, dataloader, device)

#     # Initialize variables
#     scales_index = 0
#     # exit()

#     ###########
#     # Loop over each layer and prune one layer at a time (mode == 18)

#     use_cache = model.config.use_cache 
#     model.config.use_cache = False 

#     print("loading calibdation data")
#     # trying wikitext loader
#     # dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
#     print("dataset loading complete")
#     with torch.no_grad():
#         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

#     layers = model.model.layers
#     awq_scale = torch.load('/home/tomyeh/ARCALA/wanda/awq_scales')
#     print(f"awq_scale type: {type(awq_scale)}", flush=True)
#     print(f"awq_scale length or shape: {len(awq_scale) if isinstance(awq_scale, list) else awq_scale.shape}", flush=True)
#     scales_index = 0
# ############

#     for i in range(len(layers)):

#         if mode == 17:
#             if i != int(layer_name):
#                 continue

#         layer = layers[i]
#         subset = find_layers(layer)

#         # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#         #     dev = model.hf_device_map[f"model.layers.{i}"]
#         #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = WrappedGPT(subset[name])

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 wrapped_layers[name].add_batch(inp[0].data, out.data)
#             return tmp

#         handles = []
#         for name in wrapped_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         for h in handles:
#             h.remove()

#         for name in subset:
#             print(f"pruning layer {i} name {name}")
#             # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#             if mode < 4:
#                 W_metric = torch.pow(torch.abs(subset[name].weight.data), mode) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             elif mode == 4:
#                 W_metric = torch.abs(subset[name].weight.data) 
#             elif mode == 5:
#                 if name == 'self_attn.v_proj' or name == 'mlp.gate_proj' or name == "mlp.up_proj" or name == "mlp.down_proj":
#                     # 6.357 perplexity
#                     W_metric = torch.pow(torch.abs(subset[name].weight.data), 1.75)  * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 else:
#                     # 6.357 perplexity
#                     # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                     # 6.34
#                     # W_metric = torch.pow(torch.abs(subset[name].weight.data), 1.25)  * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                     # 6.339
#                     W_metric = torch.pow(torch.abs(subset[name].weight.data), 1.26)  * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             elif mode == 6:
#                 if name == layer_name or layer_name == 'all':
#                     weights = torch.abs(subset[name].weight.data)
#                     wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                     W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power) 
#                 else:
#                     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#             elif mode == 7:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.05)
#                 elif name == 'self_attn.v_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # elif name == 'self_attn.o_proj':
#                 #     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.8)
#                 elif name == 'mlp.down_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             elif mode == 8:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 # 4:8 perplexity 7.349258899688721
#                 # 50% perplexity 6.337991237640381
#                 # if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                 #     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.05)
#                 # elif name == 'self_attn.v_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # elif name == 'self_attn.o_proj':
#                 #     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 # elif name == 'mlp.down_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # else:
#                 #     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 # perplexity 6.323913097381592
#                 # perplexity 7.350037097930908
#                 # if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                 #     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                 # elif name == 'self_attn.v_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # elif name == 'self_attn.o_proj':
#                 #     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 # elif name == 'mlp.down_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # else:
#                 #     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                 elif name == 'self_attn.v_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 elif name == 'self_attn.o_proj':
#                     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 elif name == 'mlp.down_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#             elif mode == 9:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 #wikitext perplexity 6.345602989196777
#                 # if i < 15:
#                 if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                 elif name == 'self_attn.v_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 elif name == 'self_attn.o_proj':
#                     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 elif name == 'mlp.down_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#             elif mode == 10:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1) 

#                 W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale,wanda_power) * torch.pow(awq_ind, awq_power)
            
#             elif mode == 11:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1) 

#                 W_metric = torch.pow(weights * awq_ind, weight_power) * torch.pow(wanda_scale,wanda_power) 

#             elif mode == 12:
#                 if name == layer_name or layer_name == 'all':
#                     weights = torch.abs(subset[name].weight.data)
#                     awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1) 

#                     W_metric = torch.pow(weights, weight_power) * torch.pow(awq_ind, awq_power) 
#                 else:
#                     W_metric = torch.pow(weights, 1.0) * torch.pow(awq_ind, 1.0)       

#             elif mode == 13:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#                 awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1) 

#                 # W_metric = torch.pow(weights * awq_ind, weight_power) * torch.pow(wanda_scale,wanda_power) 
#                 #wikitext perplexity 6.345602989196777
#                 # if i < 15:
#                 if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                     W_metric = torch.pow(weights* awq_ind, 1) * torch.pow(wanda_scale, 0.1)
#                 elif name == 'self_attn.v_proj':
#                     W_metric = torch.pow(weights* awq_ind, 1.75) * torch.pow(wanda_scale, 1)
#                 elif name == 'self_attn.o_proj':
#                     W_metric = torch.pow(weights* awq_ind, 1.25) * torch.pow(wanda_scale, 1)
#                 elif name == 'mlp.down_proj':
#                     W_metric = torch.pow(weights* awq_ind, 1.75) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.abs(subset[name].weight.data * awq_ind) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))                          
#                 #wikitext perplexity 6.345602989196777
#                 # if i < 15:
#                 # if name == 'mlp.gate_proj' or name == "mlp.up_proj":
#                 #     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                 # elif name == 'self_attn.v_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # elif name == 'self_attn.o_proj':
#                 #     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 # elif name == 'mlp.down_proj':
#                 #     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 # else:
#                 #     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             elif mode == 14:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 if name == 'mlp.gate_proj' or name == "mlp.up_proj": 
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                 elif name == 'self_attn.v_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 elif name == 'mlp.down_proj':
#                     W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#             elif mode == 15:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 if i == int(layer_name):
#                     if name == 'mlp.gate_proj' or name == "mlp.up_proj": 
#                         W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                     elif name == 'self_attn.v_proj':
#                         W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                     elif name == 'mlp.down_proj':
#                         W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                     else:
#                         W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 1)

#             elif mode == 16:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 if i != 19 and i != 30:
#                     if name == 'mlp.gate_proj' or name == "mlp.up_proj": 
#                         W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 0.1)
#                     elif name == 'self_attn.v_proj':
#                         W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                     elif name == 'mlp.down_proj':
#                         W_metric = torch.pow(weights, 1.75) * torch.pow(wanda_scale, 1)
#                     else:
#                         W_metric = torch.pow(weights, 1.25) * torch.pow(wanda_scale, 1)
#                 else:
#                     W_metric = torch.pow(weights, 1) * torch.pow(wanda_scale, 1)

#             # per sensitivity analysis
#             elif mode == 17:
#                 weights = torch.abs(subset[name].weight.data)
#                 wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#                 W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


#             scales_index +=1
#             W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
#             if prune_n != 0:
#                 # structured n:m sparsity
#                 for ii in range(W_metric.shape[1]):
#                     if ii % prune_m == 0:
#                         tmp = W_metric[:,ii:(ii+prune_m)].float()
#                         W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
#             else:
#                 sort_res = torch.sort(W_metric, dim=-1, stable=True)

#                 if args.use_variant:
#                     # wanda variant 
#                     tmp_metric = torch.cumsum(sort_res[0], dim=1)
#                     sum_before = W_metric.sum(dim=1)

#                     alpha = 0.4
#                     alpha_hist = [0., 0.8]
#                     W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
#                     while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
#                         if cur_sparsity > args.sparsity_ratio:
#                             alpha_new = (alpha + alpha_hist[0]) / 2.0
#                             alpha_hist[1] = alpha
#                         else:
#                             alpha_new = (alpha + alpha_hist[1]) / 2.0
#                             alpha_hist[0] = alpha

#                         alpha = alpha_new 
#                         W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
#                     print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
#                 else:
#                     # unstructured pruning
#                     indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
#                     W_mask.scatter_(1, indices, True)

#             subset[name].weight.data[W_mask] = 0  ## set weights to zero 

#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         inps, outs = outs, inps

#     model.config.use_cache = use_cache 
#     torch.cuda.empty_cache()

# updated with mode 18

def prune_wanda_new(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, mode=2, weight_power=1, wanda_power=1, awq_power=1, layer_name='all'):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    awq_scale = torch.load('/home/tomyeh/ARCALA/wanda/awq_scales')
    print(f"awq_scale type: {type(awq_scale)}", flush=True)
    print(f"awq_scale length or shape: {len(awq_scale) if isinstance(awq_scale, list) else awq_scale.shape}", flush=True)

    # Load calibration data
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    scales_index = 0

    if mode == 18:
        # Mode 18: Iteratively prune each layer and resample activations
        print("Mode 18: Iteratively pruning layers and recalculating activations...")
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # Wrap layers for collecting activations
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            # Define hook to collect activations
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Pass inputs through current layer and collect activations
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            # Remove hooks after collecting activations
            for h in handles:
                h.remove()

            # Compute W_metric using the collected activations
            for name in subset:
                print(f"pruning layer {i} name {name}")

                # Compute W_metric based on the mode and parameters
                weights = torch.abs(subset[name].weight.data)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

                if mode < 4:
                    W_metric = torch.pow(weights, mode) * wanda_scale
                elif mode == 4:
                    W_metric = weights
                elif mode == 5:
                    if name in ['self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
                        W_metric = torch.pow(weights, 1.75) * wanda_scale
                    else:
                        W_metric = torch.pow(weights, 1.26) * wanda_scale
                elif mode == 6:
                    if name == layer_name or layer_name == 'all':
                        W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
                    else:
                        W_metric = weights * wanda_scale
                elif mode == 10:
                    awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1)
                    W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power) * torch.pow(awq_ind, awq_power)
                else:
                    # You can include other modes as needed
                    W_metric = weights * wanda_scale

                scales_index += 1

                # Create the mask and prune weights
                W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize mask to all False
                if prune_n != 0:
                    # Structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # Wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
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
                        # Unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                # Set pruned weights to zero
                subset[name].weight.data[W_mask] = 0

            # Update inputs for the next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps  # Update inputs for the next iteration

    else:
        # Existing code for other modes
        for i in range(len(layers)):
            if mode == 17:
                if i != int(layer_name):
                    continue

            layer = layers[i]
            subset = find_layers(layer)

            # Wrap layers for collecting activations
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            # Define hook to collect activations
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Pass inputs through the layer to collect activations
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            # Remove hooks after collecting activations
            for h in handles:
                h.remove()

            # Compute W_metric using the collected activations
            for name in subset:
                print(f"pruning layer {i} name {name}")

                if mode < 4:
                    W_metric = torch.pow(torch.abs(subset[name].weight.data), mode) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                elif mode == 4:
                    W_metric = torch.abs(subset[name].weight.data)
                elif mode == 5:
                    if name in ['self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
                        W_metric = torch.pow(torch.abs(subset[name].weight.data), 1.75) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    else:
                        W_metric = torch.pow(torch.abs(subset[name].weight.data), 1.26) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                elif mode == 6:
                    if name == layer_name or layer_name == 'all':
                        weights = torch.abs(subset[name].weight.data)
                        wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                        W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
                    else:
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                elif mode == 10:
                    weights = torch.abs(subset[name].weight.data)
                    wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    awq_ind = awq_scale[scales_index].to(weights.device).reshape(1, -1)
                    W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power) * torch.pow(awq_ind, awq_power)
                else:
                    # You can include other modes as needed
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

                scales_index += 1

                # Create the mask and prune weights
                W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize mask to all False
                if prune_n != 0:
                    # Structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # Wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
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
                        # Unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                # Set pruned weights to zero
                subset[name].weight.data[W_mask] = 0

            # Update inputs for the next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps  # Update inputs for the next iteration

    # Restore model configuration
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_auto_search(args, model, tokenizer, device=torch.device("cuda:0"), 
                           prune_n=0, prune_m=0, layer_name='all',
                           auto_search=True, n_grid=10, 
                           weight_power_range=(0.5, 2.0), wanda_power_range=(0.5, 2.0),
                           optimize_per_layer=True, optimization_metric='mse'):
    """
    Enhanced WANDA pruning with automatic hyperparameter search.
    Similar to auto_scale.py, this finds optimal weight_power and wanda_power
    by minimizing MSE between original and pruned layer outputs.
    
    Args:
        args: Arguments object with pruning configuration
        model: Model to prune
        tokenizer: Tokenizer for data loading
        device: Device to run on
        prune_n, prune_m: Structured pruning parameters
        layer_name: Target layer name for optimization ('all' for all layers)
        auto_search: Whether to perform automatic hyperparameter search
        n_grid: Grid size for hyperparameter search
        weight_power_range: (min, max) range for weight_power search
        wanda_power_range: (min, max) range for wanda_power search
        optimize_per_layer: Whether to optimize parameters per layer (True) or globally (False)
        
    Returns:
        dict: Optimal parameters for each layer if auto_search=True, None otherwise
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("Loading calibration data...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, 
                               seqlen=model.seqlen, tokenizer=tokenizer)
    print("Dataset loading complete")
    
    # If using perplexity optimization, use a different approach
    if optimization_metric == 'perplexity':
        return prune_wanda_auto_search_perplexity(args, model, tokenizer, device, 
                                                 prune_n, prune_m, layer_name,
                                                 n_grid, weight_power_range, wanda_power_range)
    
    # Prepare calibration inputs
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    
    layers = model.model.layers
    
    # Store optimal parameters for each layer and submodule
    optimal_params = {}
    all_layer_data = []  # Store activation data for all layers
    
    if auto_search and optimize_per_layer:
        print(f"\n{'='*80}")
        print("PHASE 1: COLLECTING ACTIVATIONS AND FINDING OPTIMAL PARAMETERS")
        print(f"{'='*80}")
        
        # First pass: Collect activations and find optimal parameters for each layer
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\nCollecting activations for layer {i}/{len(layers)}")
            
            # Wrap layers for activation collection
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            
            # Collect activations
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            # Store layer data for optimization
            all_layer_data.append({
                'layer_idx': i,
                'layer': layer,
                'subset': subset,
                'wrapped_layers': wrapped_layers,
                'input_sample': inps[0].unsqueeze(0).clone()
            })
            
            # Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
        
        print(f"\n{'='*80}")
        print("PHASE 2: OPTIMIZING PARAMETERS FOR EACH LAYER")
        print(f"{'='*80}")
        
        # Second pass: Optimize parameters for each layer
        layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
        
        for layer_data in all_layer_data:
            i = layer_data['layer_idx']
            layer = layer_data['layer']
            subset = layer_data['subset']
            wrapped_layers = layer_data['wrapped_layers']
            input_sample = layer_data['input_sample']
            
            print(f"\n{'='*60}")
            print(f"Optimizing layer {i}/{len(layers)}")
            print(f"{'='*60}")
            
            # Optimize parameters for each submodule in the layer
            layer_optimal_params = {}
            
            for name in subset:
                print(f"\nOptimizing parameters for layer {i}, module {name}")
                
                best_wp, best_wd, best_error = auto_search_prune_params(
                    layer=layer,
                    subset={name: subset[name]},  # Only optimize for this specific module
                    wrapped_layers={name: wrapped_layers[name]},
                    input_batch=input_sample,
                    layer_kwargs=layer_kwargs,
                    sparsity_ratio=args.sparsity_ratio,
                    layer_name=name,  # Target this specific module
                    weight_power_range=weight_power_range,
                    wanda_power_range=wanda_power_range,
                    n_grid=n_grid,
                    optimization_metric='mse'  # Use MSE for per-layer optimization
                )
                
                layer_optimal_params[name] = {
                    'weight_power': best_wp,
                    'wanda_power': best_wd,
                    'error': best_error
                }
            
            optimal_params[i] = layer_optimal_params
        
        print(f"\n{'='*80}")
        print("PHASE 3: APPLYING OPTIMAL PARAMETERS TO ALL LAYERS")
        print(f"{'='*80}")
        
        # Third pass: Apply optimal parameters to all layers
        # Need to reset inputs and re-collect activations for final pruning
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\nApplying optimal parameters to layer {i}/{len(layers)}")
            
            # Wrap layers for activation collection
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            
            # Collect activations
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            # Apply pruning with optimal parameters for this layer
            for name in subset:
                if i in optimal_params and name in optimal_params[i]:
                    weight_power = optimal_params[i][name]['weight_power']
                    wanda_power = optimal_params[i][name]['wanda_power']
                else:
                    # Fallback to default if no optimal params found
                    weight_power, wanda_power = 1.0, 1.0
                
                print(f"Pruning layer {i} module {name} with wp={weight_power:.3f}, wd={wanda_power:.3f}")
                
                weights = torch.abs(subset[name].weight.data)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
                
                # Create and apply pruning mask
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                
                if prune_n != 0:
                    # Structured N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    # Unstructured sparsity
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if args.use_variant:
                        # Wanda variant with cumulative sum thresholding
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"Alpha found {alpha:.4f}, sparsity {cur_sparsity:.6f}")
                    else:
                        # Standard unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)
                
                subset[name].weight.data[W_mask] = 0
            
            # Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
    else:
        # Original single-pass approach for backward compatibility
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\n{'='*60}")
            print(f"Processing layer {i}/{len(layers)}")
            print(f"{'='*60}")
            
            # Wrap layers for activation collection
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            
            # Collect activations
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            if auto_search:
                # Perform automatic hyperparameter search
                layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
                
                # Use first sample as representative input for optimization
                input_sample = inps[0].unsqueeze(0)
                
                best_wp, best_wd, best_error = auto_search_prune_params(
                    layer=layer,
                    subset=subset,
                    wrapped_layers=wrapped_layers,
                    input_batch=input_sample,
                    layer_kwargs=layer_kwargs,
                    sparsity_ratio=args.sparsity_ratio,
                    layer_name=layer_name,
                    weight_power_range=weight_power_range,
                    wanda_power_range=wanda_power_range,
                    n_grid=n_grid,
                    optimization_metric='mse'  # Use MSE for per-layer optimization
                )
                
                optimal_params[i] = {
                    'weight_power': best_wp, 
                    'wanda_power': best_wd, 
                    'error': best_error,
                    'layer_name': layer_name
                }
                
                # Apply optimal parameters
                weight_power, wanda_power = best_wp, best_wd
            else:
                # Use default parameters (standard WANDA)
                weight_power, wanda_power = 1.0, 1.0
            
            # Apply pruning with determined parameters
            for name in subset:
                print(f"Pruning layer {i} module {name} with wp={weight_power:.3f}, wd={wanda_power:.3f}")
                
                if layer_name == 'all' or name == layer_name:
                    weights = torch.abs(subset[name].weight.data)
                    wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
                else:
                    # Use standard WANDA for non-target layers
                    weights = torch.abs(subset[name].weight.data)
                    wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_metric = weights * wanda_scale
                
                # Create and apply pruning mask
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                
                if prune_n != 0:
                    # Structured N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    # Unstructured sparsity
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if args.use_variant:
                        # Wanda variant with cumulative sum thresholding
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"Alpha found {alpha:.4f}, sparsity {cur_sparsity:.6f}")
                    else:
                        # Standard unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)
                
                subset[name].weight.data[W_mask] = 0
            
            # Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    if auto_search:
        # Print summary of optimal parameters
        print("\n" + "="*100)
        print("OPTIMAL PARAMETERS SUMMARY")
        print("="*100)
        
        if optimize_per_layer:
            # Per-layer, per-module summary
            print(f"{'Layer':<6} {'Module':<20} {'Weight Power':<12} {'Wanda Power':<11} {'MSE Error':<12}")
            print("-" * 100)
            for layer_idx, layer_params in optimal_params.items():
                for module_name, params in layer_params.items():
                    print(f"{layer_idx:<6} {module_name:<20} {params['weight_power']:<12.4f} "
                          f"{params['wanda_power']:<11.4f} {params['error']:<12.6f}")
        else:
            # Legacy format for single-layer optimization
            print(f"{'Layer':<6} {'Weight Power':<12} {'Wanda Power':<11} {'MSE Error':<12} {'Target Layer'}")
            print("-" * 100)
            for layer_idx, params in optimal_params.items():
                if isinstance(params, dict) and 'weight_power' in params:
                    print(f"{layer_idx:<6} {params['weight_power']:<12.4f} {params['wanda_power']:<11.4f} "
                          f"{params['error']:<12.6f} {params.get('layer_name', 'N/A')}")
        print("="*100)
        
        return optimal_params
    
    return None

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
def get_layer_type(layer_name):
    """
    Categorize layer by type based on its name.
    
    Args:
        layer_name (str): Name of the layer (e.g., 'self_attn.q_proj', 'mlp.gate_proj')
    
    Returns:
        str: Layer type category
    """
    if 'self_attn' in layer_name:
        if 'q_proj' in layer_name:
            return 'attention_q'
        elif 'k_proj' in layer_name:
            return 'attention_k'
        elif 'v_proj' in layer_name:
            return 'attention_v'
        elif 'o_proj' in layer_name:
            return 'attention_o'
        else:
            return 'attention_other'
    elif 'mlp' in layer_name:
        if 'gate_proj' in layer_name:
            return 'mlp_gate'
        elif 'up_proj' in layer_name:
            return 'mlp_up'
        elif 'down_proj' in layer_name:
            return 'mlp_down'
        else:
            return 'mlp_other'
    else:
        return 'other'


@torch.no_grad()
def auto_search_prune_params_by_layer_type(layers_data, layer_type_samples, sparsity_ratio,
                                         weight_power_range=(0.5, 2.0), wanda_power_range=(0.5, 2.0),
                                         n_grid=20):
    """
    Automatically search for optimal weight_power and wanda_power parameters for a specific layer type
    by minimizing MSE between original and pruned layer outputs across multiple representative layers.
    
    Args:
        layers_data: List of layer data containing layer objects, subsets, wrapped_layers
        layer_type_samples: Dictionary mapping layer type to list of representative samples
        sparsity_ratio: Target sparsity ratio
        weight_power_range: (min, max) range for weight_power search
        wanda_power_range: (min, max) range for wanda_power search
        n_grid: Number of grid points to search
        
    Returns:
        dict: Mapping from layer_type to optimal parameters {weight_power, wanda_power, error}
    """
    
    optimal_params_by_type = {}
    
    for layer_type, samples in layer_type_samples.items():
        if not samples:
            continue
            
        print(f"\n{'='*60}")
        print(f"Optimizing parameters for layer type: {layer_type}")
        print(f"Using {len(samples)} representative samples")
        print(f"{'='*60}")
        
        best_error = float("inf")
        best_weight_power = -1
        best_wanda_power = -1
        
        # Generate search grids
        weight_powers = torch.linspace(weight_power_range[0], weight_power_range[1], n_grid)
        wanda_powers = torch.linspace(wanda_power_range[0], wanda_power_range[1], n_grid)
        
        print(f"Searching {n_grid}x{n_grid} grid for optimal pruning parameters...")
        
        for i, weight_power in enumerate(weight_powers):
            for j, wanda_power in enumerate(wanda_powers):
                total_error = 0.0
                valid_samples = 0
                
                # Test this parameter combination on all samples of this layer type
                for sample in samples:
                    layer_idx = sample['layer_idx']
                    layer_name = sample['layer_name']
                    layer_data = layers_data[layer_idx]
                    
                    layer = layer_data['layer']
                    subset = layer_data['subset']
                    wrapped_layers = layer_data['wrapped_layers']
                    input_sample = layer_data['input_sample']
                    layer_kwargs = layer_data['layer_kwargs']
                    
                    # Store original state
                    org_sd = {k: v.cpu().clone() for k, v in layer.state_dict().items()}
                    
                    try:
                        # Get original layer output
                        with torch.no_grad():
                            org_out = layer(input_sample, **layer_kwargs)
                            if isinstance(org_out, tuple):
                                org_out = org_out[0]
                        
                        # Apply pruning with current parameters temporarily
                        original_weights = _apply_prune_with_params_temp(
                            {layer_name: subset[layer_name]}, 
                            {layer_name: wrapped_layers[layer_name]}, 
                            sparsity_ratio,
                            weight_power.item(), 
                            wanda_power.item(), 
                            layer_name
                        )
                        
                        # Get pruned output
                        with torch.no_grad():
                            pruned_out = layer(input_sample, **layer_kwargs)
                            if isinstance(pruned_out, tuple):
                                pruned_out = pruned_out[0]
                        
                        # Calculate MSE loss
                        loss = (org_out - pruned_out).float().pow(2).mean().item()
                        total_error += loss
                        valid_samples += 1
                        
                        # Restore weights for next iteration
                        for name in original_weights:
                            subset[name].weight.data.copy_(original_weights[name])
                            
                    except Exception as e:
                        print(f"Error processing sample {layer_name}: {e}")
                        # Restore state even if error occurred
                        layer.load_state_dict(org_sd)
                        continue
                    
                    # Restore original layer state
                    layer.load_state_dict(org_sd)
                
                if valid_samples > 0:
                    avg_error = total_error / valid_samples
                    
                    # Update best parameters if this is better
                    if avg_error < best_error:
                        best_error = avg_error
                        best_weight_power = weight_power.item()
                        best_wanda_power = wanda_power.item()
                
                if (i * n_grid + j + 1) % (n_grid * 2) == 0:  # Progress update
                    progress = (i * n_grid + j + 1) / (n_grid * n_grid) * 100
                    print(f"Progress: {progress:.1f}% | Current: wp={weight_power:.3f}, wd={wanda_power:.3f}, "
                          f"error={avg_error if valid_samples > 0 else 'N/A':.6f} | "
                          f"Best so far: wp={best_weight_power:.3f}, wd={best_wanda_power:.3f}, error={best_error:.6f}")
        
        optimal_params_by_type[layer_type] = {
            'weight_power': best_weight_power,
            'wanda_power': best_wanda_power,
            'error': best_error
        }
        
        print(f"\nOptimal parameters for {layer_type}:")
        print(f"  Weight Power: {best_weight_power:.4f}")
        print(f"  Wanda Power: {best_wanda_power:.4f}")  
        print(f"  Average MSE Error: {best_error:.6f}")
    
    return optimal_params_by_type


def prune_wanda_auto_search_perplexity(args, model, tokenizer, device, 
                                      prune_n=0, prune_m=0, layer_name='all',
                                      n_grid=10, weight_power_range=(0.5, 2.0), 
                                      wanda_power_range=(0.5, 2.0)):
    """
    WANDA pruning with automatic hyperparameter search using perplexity optimization.
    This performs global parameter search by evaluating full model perplexity.
    
    Args:
        args: Arguments object with pruning configuration
        model: Model to prune
        tokenizer: Tokenizer for data loading
        device: Device to run on
        prune_n, prune_m: Structured pruning parameters
        layer_name: Target layer name for optimization ('all' for all layers)  
        n_grid: Grid size for hyperparameter search
        weight_power_range: (min, max) range for weight_power search
        wanda_power_range: (min, max) range for wanda_power search
        
    Returns:
        dict: Optimal parameters found through perplexity optimization
    """
    import copy
    from .eval import eval_ppl
    
    print(f"\n{'='*80}")
    print("PERPLEXITY-BASED OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Searching {n_grid}x{n_grid} grid for optimal pruning parameters using perplexity...")
    
    # Store original model state
    original_state_dict = copy.deepcopy(model.state_dict())
    
    # Generate search grids
    weight_powers = torch.linspace(weight_power_range[0], weight_power_range[1], n_grid)
    wanda_powers = torch.linspace(wanda_power_range[0], wanda_power_range[1], n_grid)
    
    best_perplexity = float("inf")
    best_weight_power = -1
    best_wanda_power = -1
    
    print(f"Weight power range: {weight_power_range}")
    print(f"Wanda power range: {wanda_power_range}")
    print(f"Grid size: {n_grid}x{n_grid} = {n_grid*n_grid} combinations")
    print()
    
    for i, weight_power in enumerate(weight_powers):
        for j, wanda_power in enumerate(wanda_powers):
            # Restore original model
            model.load_state_dict(original_state_dict)
            
            # Create a copy of args with current parameters
            current_args = copy.deepcopy(args) 
            current_args.weight_power = weight_power.item()
            current_args.wanda_power = wanda_power.item()
            
            print(f"Testing combination {i*n_grid + j + 1}/{n_grid*n_grid}: "
                  f"weight_power={weight_power:.3f}, wanda_power={wanda_power:.3f}")
            
            # Prune model with current parameters
            try:
                prune_wanda_new(current_args, model, tokenizer, device, 
                               prune_n=prune_n, prune_m=prune_m, 
                               mode=6, weight_power=weight_power.item(), 
                               wanda_power=wanda_power.item(), awq_power=1.0, 
                               layer_name=layer_name)
                
                # Evaluate perplexity
                current_ppl = eval_ppl(current_args, model, tokenizer, device)
                
                print(f"  Perplexity: {current_ppl:.4f}")
                
                # Update best parameters if this is better (lower perplexity)
                if current_ppl < best_perplexity:
                    best_perplexity = current_ppl
                    best_weight_power = weight_power.item()
                    best_wanda_power = wanda_power.item()
                    print(f"  *** NEW BEST: ppl={best_perplexity:.4f} ***")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                # Skip this combination if it fails
                continue
            
            print()
    
    # Restore original model and apply best parameters
    model.load_state_dict(original_state_dict) 
    
    print(f"{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best parameters found:")
    print(f"  Weight Power: {best_weight_power:.4f}")
    print(f"  Wanda Power: {best_wanda_power:.4f}")
    print(f"  Best Perplexity: {best_perplexity:.4f}")
    print()
    
    # Apply the best parameters to the model
    if best_weight_power != -1:
        print("Applying best parameters to model...")
        best_args = copy.deepcopy(args)
        best_args.weight_power = best_weight_power
        best_args.wanda_power = best_wanda_power
        
        prune_wanda_new(best_args, model, tokenizer, device,
                       prune_n=prune_n, prune_m=prune_m, 
                       mode=6, weight_power=best_weight_power,
                       wanda_power=best_wanda_power, awq_power=1.0,
                       layer_name=layer_name)
        
        print("Model pruned with optimal parameters!")
    else:
        print("No valid parameters found. Model unchanged.")
    
    # Return optimal parameters in expected format
    optimal_params = {
        'global': {
            'weight_power': best_weight_power,
            'wanda_power': best_wanda_power, 
            'perplexity': best_perplexity,
            'optimization_metric': 'perplexity'
        }
    }
    
    return optimal_params


@torch.no_grad()
def prune_wanda_auto_layer_type_search(args, model, tokenizer, device=torch.device("cuda:0"), 
                                     prune_n=0, prune_m=0, auto_search=True, n_grid=10, 
                                     weight_power_range=(0.5, 2.0), wanda_power_range=(0.5, 2.0),
                                     optimization_metric='mse'):
    """
    Enhanced WANDA pruning with automatic hyperparameter search per layer type.
    This finds optimal weight_power and wanda_power for each layer type (attention_q, attention_k, etc.)
    by minimizing MSE between original and pruned layer outputs.
    
    Args:
        args: Arguments object with pruning configuration
        model: Model to prune
        tokenizer: Tokenizer for data loading
        device: Device to run on
        prune_n, prune_m: Structured pruning parameters
        auto_search: Whether to perform automatic hyperparameter search
        n_grid: Grid size for hyperparameter search
        weight_power_range: (min, max) range for weight_power search
        wanda_power_range: (min, max) range for wanda_power search
        
    Returns:
        dict: Optimal parameters for each layer type if auto_search=True, None otherwise
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print("Loading calibration data...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, 
                               seqlen=model.seqlen, tokenizer=tokenizer)
    print("Dataset loading complete")
    
    # Prepare calibration inputs
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    
    layers = model.model.layers
    
    # Store layer data and categorize by type
    all_layer_data = []
    layer_type_samples = {}  # Maps layer type to list of sample references
    
    if auto_search:
        print(f"\n{'='*80}")
        print("PHASE 1: COLLECTING ACTIVATIONS AND CATEGORIZING BY LAYER TYPE")
        print(f"{'='*80}")
        
        # First pass: Collect activations and categorize layers by type
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\nCollecting activations for layer {i}/{len(layers)}")
            
            # Wrap layers for activation collection
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            
            # Collect activations
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            # Store layer data
            layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
            layer_data = {
                'layer_idx': i,
                'layer': layer,
                'subset': subset,
                'wrapped_layers': wrapped_layers,
                'input_sample': inps[0].unsqueeze(0).clone(),
                'layer_kwargs': layer_kwargs
            }
            all_layer_data.append(layer_data)
            
            # Categorize layers by type and collect samples
            for name in subset:
                layer_type = get_layer_type(name)
                if layer_type not in layer_type_samples:
                    layer_type_samples[layer_type] = []
                
                # Add this as a sample for this layer type (limit samples per type for efficiency)
                if len(layer_type_samples[layer_type]) < 3:  # Max 3 samples per type
                    layer_type_samples[layer_type].append({
                        'layer_idx': i,
                        'layer_name': name,
                        'layer_type': layer_type
                    })
            
            # Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
        
        print(f"\n{'='*80}")
        print("PHASE 2: OPTIMIZING PARAMETERS FOR EACH LAYER TYPE")
        print(f"{'='*80}")
        
        # Print layer type summary
        print("Layer type samples collected:")
        for layer_type, samples in layer_type_samples.items():
            print(f"  {layer_type}: {len(samples)} samples")
        
        # Find optimal parameters for each layer type
        optimal_params_by_type = auto_search_prune_params_by_layer_type(
            all_layer_data, layer_type_samples, args.sparsity_ratio,
            weight_power_range=weight_power_range,
            wanda_power_range=wanda_power_range,
            n_grid=n_grid
        )
        
        print(f"\n{'='*80}")
        print("PHASE 3: APPLYING OPTIMAL PARAMETERS BY LAYER TYPE")
        print(f"{'='*80}")
        
        # Third pass: Reset model and apply optimal parameters
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\nApplying optimal parameters to layer {i}/{len(layers)}")
            
            # Wrap layers for activation collection
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            
            # Collect activations
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            # Apply pruning with optimal parameters for each layer type
            for name in subset:
                layer_type = get_layer_type(name)
                
                if layer_type in optimal_params_by_type:
                    weight_power = optimal_params_by_type[layer_type]['weight_power']
                    wanda_power = optimal_params_by_type[layer_type]['wanda_power']
                else:
                    # Fallback to default if no optimal params found
                    weight_power, wanda_power = 1.0, 1.0
                
                print(f"Pruning layer {i} module {name} (type: {layer_type}) with wp={weight_power:.3f}, wd={wanda_power:.3f}")
                
                weights = torch.abs(subset[name].weight.data)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = torch.pow(weights, weight_power) * torch.pow(wanda_scale, wanda_power)
                
                # Create and apply pruning mask
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                
                if prune_n != 0:
                    # Structured N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    # Unstructured sparsity
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if args.use_variant:
                        # Wanda variant with cumulative sum thresholding
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"Alpha found {alpha:.4f}, sparsity {cur_sparsity:.6f}")
                    else:
                        # Standard unstructured pruning
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)
                
                subset[name].weight.data[W_mask] = 0
            
            # Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
    else:
        # Standard WANDA pruning without auto-search
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
            print(f"\n{'='*60}")
            print(f"Processing layer {i}/{len(layers)} (standard WANDA)")
            print(f"{'='*60}")
            
            # Standard WANDA processing...
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
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                   position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()
            
            # Apply standard WANDA pruning (weight_power=1.0, wanda_power=1.0)
            for name in subset:
                print(f"Pruning layer {i} module {name} with standard WANDA")
                
                weights = torch.abs(subset[name].weight.data)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = weights * wanda_scale
                
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                
                if prune_n != 0:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if args.use_variant:
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"Alpha found {alpha:.4f}, sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)
                
                subset[name].weight.data[W_mask] = 0
            
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids)[0]
            inps, outs = outs, inps
    
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    if auto_search:
        # Print summary of optimal parameters by layer type
        print("\n" + "="*100)
        print("OPTIMAL PARAMETERS SUMMARY BY LAYER TYPE")
        print("="*100)
        
        print(f"{'Layer Type':<20} {'Weight Power':<12} {'Wanda Power':<11} {'MSE Error':<12}")
        print("-" * 100)
        for layer_type, params in optimal_params_by_type.items():
            print(f"{layer_type:<20} {params['weight_power']:<12.4f} {params['wanda_power']:<11.4f} "
                  f"{params['error']:<12.6f}")
        print("="*100)
        
        return optimal_params_by_type
    
    return None
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