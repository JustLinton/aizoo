import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes=91,
                 pretrained_backbone=True,
                 trainable_backbone_layers: Optional[int] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 width_mult: float = 1.0,
                 min_depth: int = 16
                 ):
        super().__init__()

        self.model = ssdlite320_mobilenet_v3_large(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                            pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers,
                                                   norm_layer=norm_layer,width_mult=width_mult,min_depth=min_depth)

    def forward(self, x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


