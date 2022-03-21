import torch.nn as nn
from typing import Dict
from torch import Tensor


class Hardshrink(nn.Hardshrink):
       
    def forward(self, input: Tensor) -> Dict:
        output = super().forward(input)
        return {
            "output": output
        }

