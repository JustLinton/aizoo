import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def group_norm(
    input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-5
)  -> Dict:
    return {
        "output": torch.nn.functional.group_norm(input,num_groups,weight,bias,eps)
    }
