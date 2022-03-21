import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def elu_(input: Tensor, alpha: float = 1.0) -> Dict:
    return {
        "output": torch.nn.functional.elu_(input,alpha)
    }
