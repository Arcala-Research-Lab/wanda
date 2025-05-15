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

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

    OUT_PATH = Path(f'/home/oyahia/yeh-research/out/raw_percents')
    layertype_to_sparsity = dict()
    for w_sparsity in tqdm([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]):
        for salient_sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
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

                    if name not in layertype_to_sparsity:
                        layertype_to_sparsity[name] = {}
                    if 'all' not in layertype_to_sparsity:
                        layertype_to_sparsity['all'] = {}
                    if w_sparsity not in layertype_to_sparsity[name]:
                        layertype_to_sparsity[name][w_sparsity] = [0, 0]
                    if w_sparsity not in layertype_to_sparsity['all']:
                        layertype_to_sparsity['all'][w_sparsity] = [0, 0]
                    layertype_to_sparsity[name][w_sparsity][0] += torch.sum(W_mask_salient)
                    layertype_to_sparsity[name][w_sparsity][1] += int(W_mask_salient.numel()*min_weights)
                    layertype_to_sparsity['all'][w_sparsity][0] += torch.sum(W_mask_salient)
                    layertype_to_sparsity['all'][w_sparsity][1] += int(W_mask_salient.numel()*min_weights)

                for j in range(args.nsamples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                inps, outs = outs, inps
        pickle.dump(layertype_to_sparsity, (OUT_PATH / f'{salient_sparsity}.pkl').open('wb'))

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()