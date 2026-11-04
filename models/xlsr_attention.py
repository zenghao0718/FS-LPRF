# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license; see LICENSES/Fairseq-MIT.txt.
# Adapted by Yixuan in Layer-Wise, commit 88efe1376dfc5f5b9b62d016d33a7381c158d8af.
# FS-LPRF extraction provenance: docs/CODE_PROVENANCE.md.
"""The Fairseq self-attention path exercised by frozen XLS-R-300M.

Preserves projection names, constructor RNG order and the original PyTorch
multi_head_attention_forward call. No decoder, pruning or xFormers path is used.
"""
import math
from typing import List, Optional
import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attention=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = self.vdim = embed_dim
        self.qkv_same_dim = True
        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        # Keep k/v/q construction and initialization in the original order.
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        assert query.size(-1) == self.embed_dim
        assert key is not None and value is not None
        return F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p,
            self.out_proj.weight, self.out_proj.bias,
            self.training or self.dropout_module.apply_during_inference,
            key_padding_mask.bool() if key_padding_mask is not None else None,
            need_weights, attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
