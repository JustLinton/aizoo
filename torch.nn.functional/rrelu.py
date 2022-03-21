import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def rrelu(
    input: Tensor, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False, inplace: bool = False
) -> Dict:
    return {
        "output": torch.nn.functional.rrelu(input,lower,upper,training,inplace)
    }
