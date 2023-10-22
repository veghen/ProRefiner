# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, args, embed_dim, edge_dim, num_heads, d_q, d_k, d_v, dropout = 0.0, bias = True):
        super().__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(d_k, embed_dim, bias = bias)
        self.v_proj = nn.Linear(d_v, embed_dim, bias = bias)
        self.q_proj = nn.Linear(d_q, embed_dim, bias = bias)

        self.h_attend = args.h_attend
        if self.h_attend:
            self.t = nn.Parameter(torch.randn(1))
        
        self.out_proj = nn.Linear(embed_dim + edge_dim, embed_dim, bias = bias)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.drop_edge = 0 if edge_dim == 0 else args.drop_edge
        if self.drop_edge > 0.:
            self.dropout_module2 = FairseqDropout(
                self.drop_edge, module_name=self.__class__.__name__
            )

        self.dense = PositionwiseFeedForward(embed_dim, embed_dim * 4)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, h_V, k, v, E = None, E_idx = None, bias = None, mask = None): 
        q = h_V.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q *= self.scaling

        tgt_len, bsz, embed_dim = q.size()
        src_len = tgt_len

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2)).view(bsz, self.num_heads, tgt_len, src_len)

        # assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if bias is not None:
            attn_weights = attn_weights + bias.view(bsz, 1, tgt_len, src_len).expand(bsz, self.num_heads, tgt_len, src_len)

        # don't attend to padding symbols
        mask_atten = mask.unsqueeze(1) if len(mask.shape) == 3 else mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(
            mask_atten == 0,
            -1e9,  #float("-inf"),
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if self.h_attend:
            attn_weights = attn_weights / (4 ** F.tanh(self.t))
        attn_weights_float = utils.softmax(attn_weights, dim = -1, onnx_trace = False)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attent_ret = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).mean(dim = 1)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if E is not None:
            mean_attend = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).mean(dim = 1)
            if E_idx is not None:
                mean_attend = torch.gather(mean_attend, -1, E_idx)
            mean_attend_edge = mean_attend.view(bsz, tgt_len, mean_attend.shape[-1], 1)

            if self.drop_edge > 0.:
                mean_attend_edge = self.dropout_module2(mean_attend_edge)
            edge_feat = (mean_attend_edge * E).sum(2) / (mean_attend_edge.sum(2) + 1e-6)

            attn = self.out_proj(torch.cat([attn.transpose(0, 1), edge_feat], dim = -1))
        else:
            attn = self.out_proj(attn.transpose(0, 1))
        h_V = self.norm(h_V + self.dropout(attn)) if h_V.shape[-1] == embed_dim else self.norm(self.dropout(attn))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        return h_V