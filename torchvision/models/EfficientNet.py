import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
# from torchvision.models import EfficientNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class EfficientNet(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool = False,
                 progress: bool = True,
                 stochastic_depth_prob: float = 0.2,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super().__init__()
        assert model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                              'efficientnet_b6', 'efficientnet_b7']
        if model_name == "efficientnet_b0":
            self.model = efficientnet_b0(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b1":
            self.model = efficientnet_b1(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b2":
            self.model = efficientnet_b2(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b3":
            self.model = efficientnet_b3(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b4":
            self.model = efficientnet_b4(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b5":
            self.model = efficientnet_b5(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b6":
            self.model = efficientnet_b6(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)
        elif model_name == "efficientnet_b7":
            self.model = efficientnet_b7(pretrained=pretrained, progress=progress, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                                  block=block,norm_layer=norm_layer)



    def forward(self, x: Tensor) -> Dict:
        output = self.model.forward(x)
        return {
            "output": output
        }


