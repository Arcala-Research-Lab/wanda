import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        ### 
        self.raw_activations = [] 
        self.capture_raw = True
        ###

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # CAPTURE RAW ACTIVATIONS FIRST, before any transformations
        if self.capture_raw and len(self.raw_activations) < 2:
            if len(inp.shape) == 2:
                inp_to_save = inp.unsqueeze(0)  # (1, seq*batch, in_features)
            else:
                inp_to_save = inp  # (batch, seq, in_features)
            self.raw_activations.append(inp_to_save.detach().cpu().clone())
            
            if len(self.raw_activations) >= 2:
                self.capture_raw = False
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
         