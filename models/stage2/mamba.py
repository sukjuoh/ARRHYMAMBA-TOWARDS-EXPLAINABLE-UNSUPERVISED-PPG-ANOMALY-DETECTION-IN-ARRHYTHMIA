
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from functools import partial
from einops import rearrange, repeat
from models.stage2.ssm import mamba_inner_fn_no_out_proj
from models.stage2.layer_norm import RMSNorm



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class BiMambaLayer(nn.Module):
    """ Optimized Bi-directional Mamba Layer """
    def __init__(self, dim, d_state=16, d_conv=3, dt_rank="auto", drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(dim, dim, bias = True)
        self.conv_1d = nn.Conv1d(dim, dim // 2, kernel_size=d_conv, bias=True, padding="same", groups=dim // 2)



        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=dim // 2).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        
        self.Ab_log = nn.Parameter(torch.log(A.clone()))
        
        self.D = nn.Parameter(torch.ones(dim // 2))
        self.x_proj = nn.Linear(dim // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim // 2, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(dim // 2) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_proj.bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        batch, seqlen, dim = x.shape

        x_proj = self.in_proj(rearrange(x, 'b (f t) c -> (b f) t c', f=3))
        x_proj = rearrange(x_proj, "b l c -> b c l")
        x_proj = self.conv_1d(x_proj)  
        
        x_proj = F.silu(x_proj)
        z_proj = x_proj.clone()
        A = -torch.exp(self.A_log.float())
        Ab = -torch.exp(self.Ab_log.float())

        forward_out = mamba_inner_fn_no_out_proj(x_proj, self.x_proj.weight, self.dt_proj.weight, A, None, None, self.D.float(),
                                                 delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        backward_out = mamba_inner_fn_no_out_proj(x_proj.flip(dims=(-1,)), self.x_proj.weight, self.dt_proj.weight, Ab, None, None, self.D.float(),
                                                  delta_bias=self.dt_proj.bias.float(), delta_softplus=True).flip(dims=(-1,))
      
        self._last_forward_out = forward_out.detach()
        self._last_backward_out = backward_out.detach()

        x_proj = forward_out + backward_out

        concat_out = torch.cat([x_proj, z_proj], dim=1)
        concat_out = rearrange(concat_out, "b c l -> b l c")

        return rearrange(self.out_proj(concat_out), '(b f) t c -> b (f t) c', f=3)


class TransitionBlock(nn.Module):
    def __init__(self, dim, expansion: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, expansion * dim)
        self.conv = nn.Conv2d(expansion * dim, expansion * dim, kernel_size=3, padding="same", groups=expansion * dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(expansion * dim, dim)

    def forward(self, x):
        x = self.proj_in(rearrange(x, 'b (f t) c -> b f t c', f=3))

        x = rearrange(x, 'b f t c -> b c f t')
        x = self.conv(x)
        x = self.activation(x)
        x = rearrange(x, 'b c f t -> b f t c')       
        return rearrange(self.proj_out(x), 'b f t c -> b (f t) c')


class BiMambaBlock(nn.Module):
    def __init__(self, dim, n_layer=1, drop_path=0.1):
        super().__init__()
        
        self.mamba_layers = nn.ModuleList([
            BiMambaLayer(dim, d_state=16, d_conv=3, drop_path=drop_path) for _ in range(n_layer)
        ])
        self.norm = RMSNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. and self.training else nn.Identity()
        self.mlp = TransitionBlock(dim, 4)

        self.apply(lambda module: init_weights(module, n_layer=n_layer))

    def forward(self, x):

        for layer in self.mamba_layers:
            x = x + self.drop_path(layer(x))
        x = x + self.drop_path(self.mlp(self.norm(x)))

        return self.norm(x)
