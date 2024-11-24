import torch

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
