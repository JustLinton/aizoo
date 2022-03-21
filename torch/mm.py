import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def mm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor]=None) -> Dict:
    return {
        "output": torch.mm(input,mat2,out=out)
    }
