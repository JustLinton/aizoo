import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Dict:
    return {
        "output": torch.nn.functional.bilinear(input1,input2,weight,bias)
    }
