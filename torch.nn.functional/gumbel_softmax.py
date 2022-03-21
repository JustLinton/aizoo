import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def gumbel_softmax(input: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Dict:
    return {
        "output": torch.nn.functional.gumbel_softmax(input,tau,hard,eps,dim)
    }
