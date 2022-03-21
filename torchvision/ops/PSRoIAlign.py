import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision

class PSRoIAlign(torchvision.ops.PSRoIAlign):
    def forward(self, input: Tensor, rois: Tensor):
        output = super().forward(input,rois)
        return {
            "output": output
        }

