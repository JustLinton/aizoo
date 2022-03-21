import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision

class FeaturePyramidNetwork(torchvision.ops.FeaturePyramidNetwork):
    def forward(self, x: Dict[str, Tensor]):
        output = super().forward(input,x)
        return {
            "output": output
        }

