import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision

class MultiScaleRoIAlign(torchvision.ops.MultiScaleRoIAlign):
    def forward(self, x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]]):
        output = super().forward(input,x,boxes,image_shapes)
        return {
            "output": output
        }

