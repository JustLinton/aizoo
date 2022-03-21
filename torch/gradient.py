import torch
from torch import Tensor
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def gradient(input: Tensor, *, spacing: Union[Tuple[Tensor, ...], List[Tensor]], dim: _size, edge_order: _int=1) -> Dict:
    return {
        "output": torch.gradient(input,spacing,dim,edge_order)
    }
