import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
import torchvision.ops as ops
# from torchvision.ops import box_convert


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Dict:
    output = ops.box_convert(boxes,in_fmt,out_fmt)
    return {
        "output": output
    }

