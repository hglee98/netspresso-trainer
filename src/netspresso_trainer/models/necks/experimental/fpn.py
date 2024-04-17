"""
This code is modified version of mmdetection.
https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py
"""
from typing import List, Dict, Type

from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F

from ...utils import BackboneOutput
from ...op.custom import ConvLayer, SeparableConvLayer

BLOCK_FROM_LITERAL: Dict[str, Type[nn.Module]] = {
    'conv': ConvLayer,
    'separable_conv': SeparableConvLayer,
}


class FPN(nn.Module):

    def __init__(
        self,
        intermediate_features_dim: List[int],
        params: DictConfig,
    ):
        super(FPN, self).__init__()

        self.in_channels = intermediate_features_dim
        self.out_channels = intermediate_features_dim[-1]
        self.num_ins = len(self.in_channels)
        self.num_outs = params.num_outs
        self.relu_before_extra_convs = params.relu_before_extra_convs
        self.add_extra_convs = params.add_extra_convs
        self.upsample_interpolation = params.upsample_interpolation

        self.lateral_conv_type = params.lateral_conv_type
        self.num_lateral_conv = params.num_lateral_conv
        self.fpn_conv_type = params.fpn_conv_type
        self.num_fpn_conv = params.num_fpn_conv
        self.norm_type = params.norm_type
        self.act_type = params.act_type

        assert self.num_outs >= self.num_ins

        self.add_extra_convs = self.add_extra_convs
        assert isinstance(self.add_extra_convs, (str, bool))
        if isinstance(self.add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert self.add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif self.add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        lateral_convlayer = BLOCK_FROM_LITERAL[self.lateral_conv_type]
        fpn_convlayer = BLOCK_FROM_LITERAL[self.fpn_conv_type]
        for i in range(self.num_ins):
            l_conv = []
            for _ in range(self.num_lateral_conv):
                l_conv += [lateral_convlayer(in_channels=self.in_channels[i], out_channels=self.out_channels,
                                             kernel_size=1, stride=1, norm_type=self.norm_type, act_type=self.act_type)]
            l_conv = nn.Sequential(*l_conv)

            fpn_conv = []
            for _ in range(self.num_fpn_conv):
                fpn_conv += [fpn_convlayer(in_channels=self.out_channels, out_channels=self.out_channels,
                                           kernel_size=3, stride=1)]
            fpn_conv = nn.Sequential(*fpn_conv)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self.num_outs - self.num_ins
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.num_ins - 1]
                else:
                    in_channels = self.out_channels
                extra_fpn_conv = fpn_convlayer(in_channels=in_channels, out_channels=self.out_channels,
                                               kernel_size=3, stride=2)
                self.fpn_convs.append(extra_fpn_conv)

        self._intermediate_features_dim = [self.out_channels for _ in range(self.num_outs)]

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode=self.upsample_interpolation)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return BackboneOutput(intermediate_features=outs)
    
    @property
    def intermediate_features_dim(self):
        return self._intermediate_features_dim


def fpn(intermediate_features_dim, conf_model_neck, **kwargs):
    return FPN(intermediate_features_dim=intermediate_features_dim, params=conf_model_neck.params)