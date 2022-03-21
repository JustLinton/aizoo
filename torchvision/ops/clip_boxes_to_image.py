import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import clip_boxes_to_image


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Dict:
    output = ops.clip_boxes_to_image(boxes,size)
    return {
        "output": output
    }
