import torch.nn as nn
from typing import Dict
from torch import Tensor


class HingeEmbeddingLoss(nn.HingeEmbeddingLoss):

    def forward(self, input: Tensor, target: Tensor)  -> Dict:
        output = super().forward(input,target)
        return {
            "output": output
        }
