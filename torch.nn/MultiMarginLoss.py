import torch.nn as nn
from typing import Dict
from torch import Tensor


class MultiMarginLoss(nn.MultiMarginLoss):

    def forward(self, input: Tensor, target: Tensor) -> Dict:
        output = super().forward(input,target)
        return {
            "output": output
        }
