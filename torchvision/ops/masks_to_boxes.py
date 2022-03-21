import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import masks_to_boxes


def masks_to_boxes(masks: torch.Tensor) -> Dict:
    output = ops.masks_to_boxes(masks)
    return {
        "output": output
    }
