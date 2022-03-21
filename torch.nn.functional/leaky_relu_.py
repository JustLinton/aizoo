import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def leaky_relu_(input: Tensor, negative_slope: float = 0.01) -> Dict:
    return {
        "output": torch.nn.functional.leaky_relu_(input,negative_slope)
    }
