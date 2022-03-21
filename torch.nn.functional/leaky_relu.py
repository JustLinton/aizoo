import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Dict:
    return {
        "output": torch.nn.functional.leaky_relu(input,negative_slope,inplace)
    }
