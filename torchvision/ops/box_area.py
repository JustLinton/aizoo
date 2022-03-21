import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import box_area

def box_area(boxes: Tensor) -> Dict:
    output = ops.box_area(boxes)
    return {
        "output": output
    }

