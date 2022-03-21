import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import ps_roi_pool

def ps_roi_pool(
    input: Tensor,
    boxes: Tensor,
    output_size: int,
    spatial_scale: float = 1.0,
)  -> Dict:
    output = ops.ps_roi_pool(input,boxes,output_size,spatial_scale)
    return {
        "output": output
    }
