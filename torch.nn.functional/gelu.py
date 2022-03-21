import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def gelu(input) -> Dict:
    return {
        "output": torch.nn.functional.gelu(input)
    }
