import torch
from torch import Tensor
from torch.nn import functional
from torch.types import _int,_size
from typing import Dict, Union, Tuple, List

def softplus(input:Tensor,beta:int=1,threshold:int=20) -> Dict:
    return {
        "output": torch.nn.functional.softplus(input,beta,threshold)
    }
