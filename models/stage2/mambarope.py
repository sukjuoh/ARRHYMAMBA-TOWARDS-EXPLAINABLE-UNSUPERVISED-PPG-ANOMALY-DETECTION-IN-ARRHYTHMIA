import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder
from typing import Optional
from functools import partial
from einops import rearrange
from models.stage2.mamba import BiMambaBlockV2
from models.stage2.transformer import SimpleTransformer
from models.stage2.layer_norm import RMSNorm

class Head(nn.Module):
    def __init__(self, H, V):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=H,
            out_channels=H,
            kernel_size=(3,3),
            padding="same",
            groups=H   
        )
        self.proj = nn.Linear(H, V)
        self.act = nn.GELU()
        self.norm = RMSNorm(V, eps=1e-4)


    def forward(self, x):   # (B,F,T,H)
        # conv operates on (B,H,F,T)
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = rearrange(self.proj(x), 'b h w c -> b (h w) c')     
        x = self.norm(self.act(x))

        return x



def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb):
    """
    :param pretrained_tok_emb: pretrained token embedding from stage 1
    :param tok_emb: token embedding of the transformer
    :return:
    """
    with torch.no_grad():
        if pretrained_tok_emb != None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb


class MambaTransformer2(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 codebook_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 mamba_layers: int,
                 depth: int,
                 heads: int,
                 attn_dim_head: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 dropout:float,
                 freq_dim: int=3,
                 **kwargs):
        """
        :param kind:
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param depth:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param pretrained_tok_emb_l: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; low-frequency
        :param pretrained_tok_emb_h: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1; high-frequency
        :param freeze_pretrained_tokens:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        self.num_tokens = num_tokens
        in_dim = embed_dim
        out_dim = embed_dim
        self.F = freq_dim
        self.dropout = dropout
        self.mask_token_idx = codebook_size
        self.mamba_layers = mamba_layers
        self.mamba_layer = BiMambaBlockV2(in_dim, self.mamba_layers)


        # token embeddings
        self.tok_emb = nn.Embedding(codebook_size+1, embed_dim)  # `+1` is for mask-token
        self.pos_emb = nn.Embedding(codebook_size+1, embed_dim)  # `+1` is for mask-token

        # transformer
        #self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.transformer = SimpleTransformer(depth=depth, embedding_dim=embed_dim, num_heads=heads, H=self.F, dropout=dropout)
        '''
        self.pred_head = nn.Sequential(*[
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        ])
        '''
    
        self.pred_head = Head(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size+1))


    def forward(self, s_M:torch.LongTensor):
        """
        s_M: (b n)
        """
        b, n = s_M.shape

        token_embeddings = self.tok_emb(s_M)  # (b n dim)
        if self.training:
            mask_ind = (s_M == self.mask_token_idx)[:,:,None]  # (b n 1)
            token_embeddings_dropout = F.dropout(token_embeddings, p=self.dropout)  # (b n d)
            token_embeddings = torch.where(mask_ind, token_embeddings, token_embeddings_dropout)  # (b n d)
        
        position_embeddings = self.pos_emb.weight[:n, :]
        
        embed = token_embeddings + position_embeddings

        #embed = token_embeddings + time_embedding + freq_embedding # (b, n, dim)
        embed = self.mamba_layer(embed)  # (b, n, dim)
        skip_con = embed.clone()
        embed = self.transformer(rearrange(embed, 'b (f t) c -> (b f) t c', f=self.F))  # (b, n, dim)
        embed = rearrange(embed, '(b f) t c -> b (f t) c', f=self.F)
        #embed = rearrange(embed, "b (h w) c -> b c h w", h=3, w=24)
        embed = self.pred_head(rearrange(embed+skip_con, 'b (f t) c -> b f t c', f=self.F))  # (b n d)
        #embed = rearrange(embed, "b c h w -> b (h w) c")
        
        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:,:,:-1]  # (b n k)
        return logits  # (b n k)
