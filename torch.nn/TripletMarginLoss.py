import torch.nn as nn
from typing import Dict
from torch import Tensor


class TripletMarginLoss(nn.TripletMarginLoss):

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Dict:
        output = super().forward(anchor,positive,negative)
        return {
            "output": output
        }
