import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def softmin(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Dict:
    return {
        "output": torch.nn.functional.softmin(input,dim,_stacklevel,dtype)
    }
