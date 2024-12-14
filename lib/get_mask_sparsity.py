import sys
import torch
from pathlib import Path
from tqdm import tqdm

if len(sys.argv) <= 1:
    print('Please specify path to 1+ mask files')
    print('Usage: python export_scale_list.py [path_to_mask1] [path_to_mask2] ...')
paths = sys.argv[1:]
for path in paths:
    path = Path(path)
    mask = torch.load(path)
    s = 0
    n = 0
    for m in mask:
        s += torch.sum(m == 1)
        n += m.numel()
    print(f'[{path}] zero: {s}, total: {n}, sparsity: {(s/n).item()}')
    

