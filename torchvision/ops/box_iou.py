import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import box_iou


def box_iou(boxes1: Tensor, boxes2: Tensor)  -> Dict:
    output = ops.box_iou(boxes1,boxes2)
    return {
        "output": output
    }

