import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.detection import ssd300_vgg16
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class SSDFeatureExtractorVGG(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes=91,
                 pretrained_backbone=True,
                 trainable_backbone_layers: Optional[int] = None,
                 ):
        super().__init__()

        self.model = ssd300_vgg16(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                            pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers)


    def forward(self, x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


