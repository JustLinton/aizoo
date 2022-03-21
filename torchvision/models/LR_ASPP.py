import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls

class LR_ASPP(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes: int = 21,
                 aux_loss: Optional[bool] = None,
                 ):
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        aux_loss=aux_loss)

    def forward(self,input: Tensor) -> Dict:
        output = self.model.forward(input)
        return {
            "output": output
        }

