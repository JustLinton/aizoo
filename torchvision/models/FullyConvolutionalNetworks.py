import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.segmentation import fcn_resnet50,fcn_resnet101
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class FullyConvolutionalNetworks(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes: int = 21,
                 aux_loss: Optional[bool] = None,
                 ):
        super().__init__()
        assert model_name in ['fcn_resnet50','fcn_resnet101','deeplabv3_mobilenet_v3_large']
        if model_name == "fcn_resnet50":
            self.model = fcn_resnet50(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        aux_loss=aux_loss)
        elif model_name == "fcn_resnet101":
            self.model = fcn_resnet101(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        aux_loss=aux_loss)

    def forward(self,x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


