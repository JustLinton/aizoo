import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List, Optional

def triu(input: Tensor, diagonal: _int=0, *, out: Optional[Tensor]=None) -> Dict:
    return {
        "output": torch.triu(input,diagonal,out=out)
    }
