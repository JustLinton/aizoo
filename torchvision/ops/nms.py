import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import nms


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Dict:
    output = ops.nms(boxes,scores,iou_threshold)
    return {
        "output": output
    }
