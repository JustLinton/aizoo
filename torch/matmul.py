import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None) -> Dict:
    return {
        "output": torch.matmul(input,other,out=out)
    }
