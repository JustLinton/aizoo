import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import ps_roi_align

def ps_roi_align(
        input: Tensor,
        boxes: Tensor,
        output_size: int,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
) -> Dict:
    output = ops.ps_roi_align(input,boxes,output_size,spatial_scale,sampling_ratio)
    return {
        "output": output
    }
