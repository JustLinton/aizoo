import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision

class StochasticDepth(torchvision.ops.StochasticDepth):
    def forward(self, input: Tensor):
        output = super().forward(input)
        return {
            "output": output
        }

