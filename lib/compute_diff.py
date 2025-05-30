import pickle
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from tqdm import tqdm

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

Ws = []

if __name__ == '__main__':
    MODEL_1 = "out/awq_models/awqw2q32good"
    MODEL_2 = "out/awq_models/awqw2q32bad"
    model_1 = get_llm(MODEL_1)
    model_2 = get_llm(MODEL_2)

    use_cache = model_1.config.use_cache 
    model_1.config.use_cache = False 
    use_cache = model_2.config.use_cache 
    model_2.config.use_cache = False 

    model1_layers = model_1.model.layers
    model2_layers = model_2.model.layers
    total_params = 0
    for i in tqdm(range(len(model1_layers))):
        model1_layer = model1_layers[i]
        model2_layer = model2_layers[i]
        subset1 = find_layers(model1_layer)
        subset2 = find_layers(model2_layer)

        for name in subset1:
            W1 = subset1[name].weight.data
            W2 = subset2[name].weight.data
            Ws.append((i, name, W1-W2))
            print(i, name, W1-W2)

    model_1.config.use_cache = use_cache 
    model_2.config.use_cache = use_cache

    pickle.dump(Ws, open('diff', 'wb'))