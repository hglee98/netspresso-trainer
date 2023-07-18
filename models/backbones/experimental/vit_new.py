"""
Based on the vit implementation of apple/ml-cvnets.
https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/models/classification/vit.py
"""
import argparse
from typing import Union, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

from models.op.ml_cvnets import ConvLayer, LinearLayer
from models.op.ml_cvnets import SinusoidalPositionalEncoding
from models.op.base_metaformer import MetaFormer, MetaFormerBlock, MetaFormerEncoder, MultiHeadSelfAttention

__all__ = ['vit']
SUPPORTING_TASK = ['classification']

class ViTEmbeddings(nn.Module):
    def __init__(self, image_channels, patch_size, hidden_size, hidden_dropout_prob, use_cls_token=True, vocab_size=1000):
        
        image_channels = 3  # {RGB}
        
        kernel_size = patch_size
        if patch_size % 2 == 0:
            kernel_size += 1
        
        self.patch_emb = ConvLayer(
            opts=None,
            in_channels=image_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=True,
            use_norm=False,
            use_act=False,
        )
        
        self.cls_token = None
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, hidden_size)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        self.pos_embed = SinusoidalPositionalEncoding(
                d_model=hidden_size,
                channels_last=True,
                max_len=vocab_size,
        )
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
    
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]  # B x 3(={RGB}) x H x W

        patch_emb = self.patch_emb(x)  # B x C(=embed_dim) x H'(=patch_size) x W'(=patch_size)
        patch_emb = patch_emb.flatten(2)  # B x C x H'*W'
        patch_emb = patch_emb.transpose(1, 2).contiguous()  # B x H'*W' x C

        # add classification token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # B x 1 x C
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)  # B x (H'*W' + 1) x C

        patch_emb = self.pos_embed(patch_emb)  # B x (H'*W' + 1) x C
        
        patch_emb = self.dropout(patch_emb)  # B x (H'*W' + 1) x C
        return patch_emb  # B x (H'*W' + 1) x C

class ViTChannelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        self.pre_norm_ffn = nn.ModuleList([
            LinearLayer(in_features=hidden_size, out_features=intermediate_size, bias=True),
            nn.SiLU(inplace=False),
            LinearLayer(in_features=intermediate_size, out_features=hidden_size, bias=True),
            nn.Dropout(p=hidden_dropout_prob),
        ])
    
    def forward(self, x):
        for layer in self.pre_norm_ffn:
            x = layer(x)
        return x

class ViTBlock(MetaFormerBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.token_mixer = MultiHeadSelfAttention(hidden_size, num_attention_heads,
                                                  attention_scale=(hidden_size // num_attention_heads) ** -0.5,
                                                  attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                  use_qkv_bias=True
                                                  )
        self.channel_mlp = ViTChannelMLP(hidden_size, intermediate_size, hidden_dropout_prob)
    
    
class ViTEncoder(MetaFormerEncoder):
    def __init__(self, num_blocks, hidden_size, num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_size, num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps) for _ in range(num_blocks)]
        )

class VisionTransformer(MetaFormer):
    def __init__(
        self,
        task,
        image_channels,
        patch_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_dropout_prob,
        layer_norm_eps=1e-6,
        use_cls_token=True,
        vocab_size=1000
    ) -> None:
        super().__init__()
        self.task = task
        self.intermediate_features = self.task in ['segmentation', 'detection']
        self.patch_embed = ViTEmbeddings(image_channels, patch_size, hidden_size, hidden_dropout_prob, use_cls_token=use_cls_token, vocab_size=vocab_size)
        self.encoder = ViTEncoder(num_hidden_layers, hidden_size, num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)


def vit(task, *args, **kwargs):
    return VisionTransformer(task, opts=None)