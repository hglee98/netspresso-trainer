# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from typing import List, Union, Tuple 
from omegaconf import DictConfig
import torch 
import torch.nn as nn 
import math

from ....op.custom import SeparableConvLayer
from ....utils import AnchorBasedDetectionModelOutput 


class YOLOFastestHeadV2(nn.Module): 
    def __init__(
        self,
        num_classes: int, 
        intermediate_features_dim: List[int], 
        params: DictConfig) -> None:
        super().__init__()
        anchors: List[Union[int, float]] = params.anchors
        num_anchors: int = len(anchors[0]) // 2
        anchor_tensor = torch.FloatTensor(anchors).view(len(anchors), num_anchors, 2) 
        anchor_tensor = anchor_tensor.unsqueeze(2).unsqueeze(2) 
        in_channel = intermediate_features_dim[0]
        self.cls_head = YOLOFastestClassificationHead(in_channel, num_anchors, num_classes)  
        self.reg_head = YOLOFastestRegressionHead(in_channel, num_anchors) 
        self.register_buffer("anchors", anchor_tensor)

    def forward(self, x: List) -> AnchorBasedDetectionModelOutput: 
        cls_logits, objs = self.cls_head(x) 
        bbox_regression = self.reg_head(x)
        preds = list() 
        for layer_idx in range(len(x)): 
            bbox, obj, cls_logit = bbox_regression[layer_idx], objs[layer_idx], cls_logits[layer_idx] 
            pred = torch.cat([bbox, obj, cls_logit], dim=-1)
            preds.append(pred)

        return AnchorBasedDetectionModelOutput(pred=preds, anchors=self.anchors)

def yolo_fastest_head_v2(num_classes, intermediate_features_dim, conf_model_head) -> YOLOFastestHeadV2:
    return YOLOFastestHeadV2(num_classes=num_classes, 
                             intermediate_features_dim=intermediate_features_dim,
                             params=conf_model_head.params)


class YOLOFastestClassificationHead(nn.Module):
    def __init__(
        self, 
        in_channels,
        num_anchors, 
        num_classes,
        prior_prob = 0.01, 
        ) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(*[
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
        ])
        self.layer_2 = nn.Sequential(*[
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
        ])
        
        self.cls_logits = nn.Conv2d(in_channels, num_classes * num_anchors, 1, 1, 0, bias=True)
        self.obj = nn.Conv2d(in_channels, num_anchors, 1, 1, 0, bias=True) 
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_prob) / prior_prob))

        self.num_classes = num_classes 
        self.num_anchors = num_anchors
    
    def forward(self, x: List) -> Tuple[List[torch.Tensor], List[torch.Tensor]]: 
        all_cls_logits = [] 
        all_objs = []
        out1 = self.layer_1(x[0])
        out2 = self.layer_2(x[1])
        outputs = [out1, out2]

        for o in outputs: 
            cls_logits = o
            objectness = cls_logits
            cls_logits = self.cls_logits(cls_logits)
            objectness = self.obj(objectness)
            
            # Permute classification output from (N, A * K, H, W) to (N, A, H, W, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 1, 3, 4, 2)

            all_cls_logits.append(cls_logits)

            # Permute objectness output from (N, A, H, W) to (N, A, H, W, 1).
            N, _, H, W = objectness.shape
            objectness = objectness.view(N, -1, 1, H, W)
            objectness = objectness.permute(0, 1, 3, 4, 2) 
            all_objs.append(objectness)
        
        return all_cls_logits, all_objs



class YOLOFastestRegressionHead(nn.Module): 
    def __init__(
        self,
        in_channels, 
        num_anchors,) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(*[
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
        ])
        self.layer_2 = nn.Sequential(*[
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
            SeparableConvLayer(in_channels, in_channels, 5, padding=2, no_out_act=True),
        ])
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, 1, 1, 0, bias=True)
    
    def forward(self, x): 
        all_bbox_regression = []
        out1 = self.layer_1(x[0])
        out2 = self.layer_2(x[1])
        outputs = [out1, out2]

        for idx, features in enumerate(x): 
            bbox_regression = outputs[idx]
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, A, H, W, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 1, 3, 4, 2)

            all_bbox_regression.append(bbox_regression)
        
        return all_bbox_regression

