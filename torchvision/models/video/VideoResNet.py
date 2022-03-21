import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.video import r3d_18,mc3_18,r2plus1d_18
# from torchvision.models import ResNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class VideoResNet(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes: int = 400,
                 zero_init_residual: bool = False
                 ):
        super().__init__()
        assert model_name in ['resnet18', 'mc3_18', 'r2plus1d_18']

        if model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained, progress=progress, num_classes=num_classes, zero_init_residual=zero_init_residual)
        elif model_name == "mc3_18":
            self.model = mc3_18(pretrained=pretrained, progress=progress, num_classes=num_classes, zero_init_residual=zero_init_residual)
        elif model_name == "r2plus1d_18":
            self.model = r2plus1d_18(pretrained=pretrained, progress=progress, num_classes=num_classes, zero_init_residual=zero_init_residual)



    def forward(self, x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


# model = ResNet("resnet18", True, True)
# # model2 = resnet18(True, True)
# input = torch.randn(5, 3, 64, 64)
# output = model(input)

