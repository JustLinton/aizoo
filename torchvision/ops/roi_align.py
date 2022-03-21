import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import roi_align


def roi_align(
        input: Tensor,
        boxes: Union[Tensor, List[Tensor]],
        output_size: BroadcastingList2[int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = False,
) -> Dict:
    output = ops.roi_align(input,boxes,output_size,spatial_scale,sampling_ratio,aligned)
    return {
        "output": output
    }
