import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Type
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from models.stage2.layer_norm import RMSNorm

"""
ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.pys
"""

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, self_attn=True, rope_dim=None, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.self_attn = self_attn

        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads

        rope_dim = rope_dim if rope_dim is not None else self.head_dim
        self.rotary_emb = RotaryEmbedding(dim=rope_dim)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: (B, L, C)
        b, l, c = x.shape
        x = x.reshape(b, l, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)   # (B, H, L, D)

    def merge_heads(self, x):
        # x: (B, H, L, D)
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, l, h * d)
        return x

    def forward(self, q: Tensor, k: Tensor = None, v: Tensor = None):
        if self.self_attn:
            k = v = q

        # projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # split heads → (B, H, L, D)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # RoPE
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # scaled dot product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        # combine heads
        out = self.merge_heads(out)

        out = self.out_proj(out)     # (B,L,C)
        out = self.dropout(out)
        return out



class RoFormer(nn.Module):
    def __init__(self, embedding_dim, num_heads, H:int, dropout=0.0):
        super().__init__()
        self.H = H
        self.self_attn = Attention(embedding_dim, num_heads, dropout=dropout)
        self.conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding='same', groups=embedding_dim)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        attn = self.self_attn(q=x)
        attn = rearrange(attn, '(b h) w c -> b c h w', h=self.H)
        attn = rearrange(self.conv(attn), 'b c h w -> (b h) w c')
        x = x + self.drop(attn)

        return x



class SimpleTransformer(nn.Module):
    def __init__(self, depth, embedding_dim, num_heads, H:int, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(embedding_dim*4, num_heads, H, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embedding_dim*4, eps=1e-5)
        self.linear2 = nn.Linear(embedding_dim*4, embedding_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x) 
        return self.linear2(x)

