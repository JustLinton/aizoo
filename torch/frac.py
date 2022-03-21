import torch
from torch import Tensor
from typing import Optional, Dict


def frac(input: Tensor, *, out: Optional[Tensor]=None) -> Dict:
    return {
        "output": torch.frac(input, out=out)
    }
