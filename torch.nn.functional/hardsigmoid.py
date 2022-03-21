import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def hardsigmoid(input: Tensor, inplace: bool = False) -> Dict:
    return {
        "output": torch.nn.functional.hardsigmoid(input,inplace)
    }
