import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def elu(input: Tensor, dim: int = -1) -> Dict:
    return {
        "output": torch.nn.functional.glu(input,dim)
    }
