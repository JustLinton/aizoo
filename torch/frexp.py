import torch
from torch import Tensor
from typing import Dict, Union, Tuple, List

def frexp(input: Tensor, *, out: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]=None) -> Dict:
    return {
        "output": torch.frexp(input, out=out)
    }
