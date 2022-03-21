import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import stochastic_depth

def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Dict:
    output = ops.stochastic_depth(input,p,mode,training)
    return {
        "output": output
    }
