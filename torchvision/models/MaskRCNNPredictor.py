import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.segmentation import fcn_resnet50
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class MaskRCNNPredictor(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes=91,
                 pretrained_backbone=True,
                 trainable_backbone_layers=None
                 ):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers)

    def forward(self,images, targets=None) -> Dict:
        output = self.model.forward(images,targets)
        return {
            "output": output
        }
