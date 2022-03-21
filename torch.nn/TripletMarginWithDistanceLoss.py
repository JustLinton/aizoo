import torch.nn as nn
from typing import Dict
from torch import Tensor


class TripletMarginWithDistanceLoss(nn.TripletMarginWithDistanceLoss):

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Dict:
        output = super().forward(anchor,positive,negative)
        return {
            "output": output
        }
