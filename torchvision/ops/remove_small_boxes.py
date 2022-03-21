import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import remove_small_boxes


def remove_small_boxes(boxes: Tensor, min_size: float) -> Dict:
    output = ops.remove_small_boxes(boxes,min_size)
    return {
        "output": output
    }
