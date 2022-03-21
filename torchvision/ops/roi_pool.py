import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import roi_pool

def roi_pool(
        input: Tensor,
        boxes: Union[Tensor, List[Tensor]],
        output_size: BroadcastingList2[int],
        spatial_scale: float = 1.0,
) -> Dict:
    output = ops.roi_pool(input,boxes,output_size,spatial_scale)
    return {
        "output": output
    }
