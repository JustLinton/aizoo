import torch
from torch import Tensor
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None) -> Dict:
    return {
        "output": torch.norm(input,p,dim,keepdim,out,dtype)
    }
