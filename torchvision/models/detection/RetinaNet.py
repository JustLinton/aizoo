import torch.nn as nn
import torch
from typing import Dict, Type, Union, List, Optional, Callable, Any
from torch import Tensor
from torchvision.models.detection import retinanet_resnet50_fpn
# from torchvision.models import RegNet
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, model_urls


class RetinaNet(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = True,
                 num_classes=91,
                 pretrained_backbone=True,
                 trainable_backbone_layers=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000
                 ):
        super().__init__()

        self.model = retinanet_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes,
                                            pretrained_backbone=pretrained_backbone,trainable_backbone_layers=trainable_backbone_layers,
                                            min_size=min_size,max_size=max_size,image_mean=image_mean,image_std=image_std,
                                            anchor_generator=anchor_generator,head=head,proposal_matcher=proposal_matcher,
                                            score_thresh=score_thresh,nms_thresh=nms_thresh,detections_per_img=detections_per_img,
                                            fg_iou_thresh=fg_iou_thresh,bg_iou_thresh=bg_iou_thresh,topk_candidates=topk_candidates)


    def forward(self, images: List[Tensor], targets: List[Dict[Tensor]]) -> Dict:
        output = self.model.forward(images,targets=None)
        #output: List[Dict[str, Tensor]]
        return {
            "output": output
        }


