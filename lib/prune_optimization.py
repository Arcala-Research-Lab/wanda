# pylint: disable=consider-using-enumerate
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .prune import prepare_calibration_input, find_layers
import torch
from scipy.optimize import minimize
from copy import deepcopy

def set_submodule(module, path, new_module):
    """
    Replaces a nested submodule in a model.
    For example, path = "attn.q_proj" will replace module.attn.q_proj
    """
    parts = path.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    setattr(module, parts[-1], new_module)

def prune_opitimize(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

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
            def optimize_raw(x, reset: bool = False) -> float:
                weights = torch.abs(subset[name].weight.data)
                wanda_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                W_metric = torch.pow(weights, x[0])  * torch.pow(wanda_scale, x[1])

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
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                old_weight = deepcopy(subset[name].weight.data)
                old_outs = [None for _ in range(len(outs))]
                for j in range(args.nsamples):
                    with torch.no_grad():
                        old_outs[j] = subset[name](inps[j].unsqueeze(0))[0]
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 
                new_outs = [None for _ in range(len(outs))]
                for j in range(args.nsamples):
                    with torch.no_grad():
                        new_outs[j] = subset[name](inps[j].unsqueeze(0))[0]
                if reset:
                    subset[name].weight.data = old_weight

                mean_val = 0
                for j in range(args.nsamples):
                    mean_val += torch.mean(torch.pow(new_outs[j] - old_outs[j], 2))/args.nsamples
                print(f"{x}, {mean_val}", end=", ")
                return mean_val.item()
            def optimize(x) -> float:
                return optimize_raw(x, reset=True)
            x0 = [1, 1]
            class Counter:
                def __init__(self) -> None:
                    self.count = 0
                def print_count(self, xk):
                    print(' iteration', self.count)
                    self.count += 1
            counter = Counter()
            result = minimize(optimize, x0, method='Powell', bounds=[(0.01, 10), (0.01, 10)], options={'maxiter': 10}, callback=counter.print_count)['x']
            print('final x:', x0)
            optimize_raw(result)


        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()