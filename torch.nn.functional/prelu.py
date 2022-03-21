import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def prelu(input: Tensor, weight: Tensor) -> Dict:
    return {
        "output": torch.nn.functional.prelu(input,weight)
    }
