import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf,regnet_y_32gf, regnet_x_400mf,regnet_x_800mf,regnet_x_1_6gf,regnet_x_3_2gf,regnet_x_8gf,regnet_x_16gf,regnet_x_32gf
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class RegNet(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes: int = 1000,
                 stem_width: int = 32,
                 stem_type: Optional[Callable[..., nn.Module]] = None,
                 block_type: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation: Optional[Callable[..., nn.Module]] = None,
                 ):
        super().__init__()
        assert model_name in ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf',
                              'regnet_y_32gf', 'regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf','regnet_x_8gf','regnet_x_16gf','regnet_x_32gf']
        if model_name == "regnet_y_400mf":
            self.model = regnet_y_400mf(pretrained=pretrained, progress=progress, num_classes=num_classes,stem_type=stem_type,stem_width=stem_width,block_type=block_type,
                                        norm_layer=norm_layer,activation=activation)
        elif model_name == "regnet_y_800mf":
            self.model = regnet_y_800mf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_y_1_6gf":
            self.model = regnet_y_1_6gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_y_3_2gf":
            self.model = regnet_y_3_2gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_y_8gf":
            self.model = regnet_y_8gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_y_16gf":
            self.model = regnet_y_16gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_y_32gf":
            self.model = regnet_y_32gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_400mf":
            self.model = regnet_x_400mf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_800mf":
            self.model = regnet_x_800mf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_1_6gf":
            self.model = regnet_x_1_6gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_3_2gf":
            self.model = regnet_x_3_2gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_8gf":
            self.model = regnet_x_8gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_16gf":
            self.model = regnet_x_16gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)
        elif model_name == "regnet_x_32gf":
            self.model = regnet_x_32gf(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                        stem_type=stem_type, stem_width=stem_width, block_type=block_type,
                                        norm_layer=norm_layer, activation=activation)

    def forward(self, x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


