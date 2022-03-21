import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn,fasterrcnn_mobilenet_v3_large_320_fpn,fasterrcnn_mobilenet_v3_large_fpn
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class FastRCNNPredictor(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes=91,
                 pretrained_backbone=True,
                 trainable_backbone_layers=None
                 ):
        super().__init__()
        assert model_name in ['fasterrcnn_resnet50_fpn','fasterrcnn_mobilenet_v3_large_320_fpn','fasterrcnn_mobilenet_v3_large_fpn']
        if model_name == "fasterrcnn_resnet50_fpn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers)
        elif model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers)
        elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
            self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x: Tensor) -> Dict:
        scores, bbox_deltas = self.model.forward(x)
        return {
            "scores": scores,
            "bbox_deltas": bbox_deltas
        }


