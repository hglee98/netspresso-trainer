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

from typing import Literal

import torch.nn as nn
import re
from omegaconf import DictConfig

from .registry import OPTIMIZER_DICT

def get_layerwise_params(model, optimizer_conf):
    default_lr = optimizer_conf.lr
    default_weight_decay = optimizer_conf.weight_decay

    param_groups = [
        {
            'params': [],
            'lr': default_lr / 10,
            'weight_decay': default_weight_decay,
            'name': 'backbone_non_norm'
        },
        {
            'params': [],
            'lr': default_lr / 10,
            'weight_decay': 0.,
            'name': 'backbone_norm'
        },
        {
            'params': [],
            'lr': default_lr,
            'weight_decay': 0.,
            'name': 'neck_head_norm'
        },
        {
            'params': [],
            'lr': default_lr,
            'weight_decay': default_weight_decay,
            'name': 'other'
        }
    ]

    # reg patterns
    patterns = [
        (r'^(?=.*backbone)(?!.*(?:norm|bn)).*$', 0),
        (r'^(?=.*backbone)(?=.*(?:norm|bn)).*$', 1),
        (r'^(?=.*(?:neck|head))(?=.*(?:norm|bn|bias)).*$', 2)
    ]

    for name, param in model.named_parameters():
        assigned = False
        for pattern, group_idx in patterns:
            if re.match(pattern, name):
                param_groups[group_idx]['params'].append(param)
                assigned = True
                break
        if not assigned:
            param_groups[3]['params'].append(param)

    param_groups = [g for g in param_groups if g['params']]
    return param_groups

def build_optimizer(
    model_or_params,
    optimizer_conf: DictConfig,
):
    parameters = model_or_params.parameters() if isinstance(model_or_params, nn.Module) else model_or_params

    opt_name: Literal['sgd', 'adam', 'adamw', 'adamax', 'adadelta', 'adagrad', 'rmsprop'] = optimizer_conf.name.lower()
    assert opt_name in OPTIMIZER_DICT
    if getattr(optimizer_conf, 'params') is not None:
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        params_groups = get_layerwise_params(model_or_params, optimizer_conf)
        optimizer = OPTIMIZER_DICT[opt_name](params_groups, optimizer_conf)
    else:
        optimizer = OPTIMIZER_DICT[opt_name](parameters, optimizer_conf)
    return optimizer
