import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def softshrink(input: Tensor, lambd: float=0.5) -> Dict:
    return {
        "output": torch.nn.functional.softshrink(input,lambd)
    }
