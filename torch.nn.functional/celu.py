import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Dict:
    return {
        "output": torch.nn.functional.celu(input,alpha,inplace)
    }
