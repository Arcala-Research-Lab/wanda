import torch
from itertools import accumulate

def awq_mask(scale_ret: list[torch.Tensor], prune_list: list[torch.Tensor], threshold: float):
    """
    scale_ret is the return value of run_awq()'s dictionary when you access the key 'scale'
    prune_list is used to see how much to repeat each row to be consistent with pruning 
    """
    awq_list = []
    pl_index = 0
    for elem in scale_ret:
        # each value is a tuple of input module, output modules, and the scale list
        ins, outs, scale_list = elem
        scale_list = scale_list.cuda() # to make topk work with Half
        # get the top 30% cutoff to emulate weights kept during pruning
        # topk_values, _ = torch.topk(scale_list, int(scale_list.numel()*0.3))
        # threshold = topk_values.min()
        # convert to boolean mask
        boolean_mask = torch.logical_not(scale_list >= threshold)
        # since AWQ goes by col, we need to repeat for each row
        repeated_mask = boolean_mask.unsqueeze(1).repeat(prune_list[pl_index].shape[0], 1)
        # since AWQ repeats Q,K,V we need to separate it out for the actual list 
        # when comparing to wanda and mag
        for _ in outs:
            awq_list.append(repeated_mask)
            pl_index += 1
    return awq_list

def get_thresholds(scales, quantiles: list[float], prune_list: list[torch.Tensor]):
    # get scales
    tensors = [scales[i][2] for i in range(len(scales))]
    # get number of weights for each scale channel
    weights = [len(prune_list[i]) for i in range(len(prune_list))]
    # 216 layers in wanda, 32 layers in awq, so need to combine layers in mag list
    indices = [0] + [j for j in accumulate(len(scales[i][1]) for i in range(len(scales)-1))]
    weights = [weights[index] for index in indices]
    # multiply the number of weights by the amount combined
    weights = [torch.tensor(weights[i]*len(scales[i][1]), dtype=torch.int64).repeat(scales[i][2].numel()) for i in range(len(scales))]
    # concatenate values and weights which are lists into one tensor
    all_values = torch.cat(tensors)
    all_weights = torch.cat(weights)

    def weighted_quantile(values, weights, quantile):
        # sort values and weights by values since quantile is based on values
        sorted_indices = torch.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # get cumulative weights to get the weighted quantile
        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        val = quantile*cumulative_weights[-1]
        quantile_index = torch.searchsorted(cumulative_weights, val)
        # find the value associated with that weighted quantile
        return sorted_values[quantile_index]

    thresholds = [weighted_quantile(all_values, all_weights, quantile).item() for quantile in quantiles]
    return thresholds