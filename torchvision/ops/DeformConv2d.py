import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision

class DeformConv2d(torchvision.ops.DeformConv2d):
    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None):
        output = super().forward(input,input,offset,mask)
        return {
            "output": output
        }

