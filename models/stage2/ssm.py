

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange, repeat

import selective_scan_cuda




class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()

        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True

        out, x = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus)

        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)

        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)

        return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x,ctx.delta_softplus
        )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

        return (
            du, ddelta, dA, dB, dC,
            dD if D is not None else None,
            ddelta_bias if delta_bias is not None else None,
            None  # `delta_softplus`의 gradient 없음
        )


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state)



def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)




class MambaInnerFnNoOutProj(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):

        assert checkpoint_lvl in [0, 1]
        L = x.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())

        if x.stride(-1) != 1:
            x = x.contiguous()

        x = x.to(dtype=torch.float32)

        x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        delta = delta.contiguous().to(dtype=torch.float32)

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None

        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous().to(dtype=torch.float32)

        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous().to(dtype=torch.float32)

        if D is not None:
            D = D.contiguous().to(dtype=torch.float32)

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)
        C = C.to(dtype=torch.float32)
        D = D.to(dtype=torch.float32)


        out, scan_intermediates = selective_scan_cuda.fwd(
            x, delta, A, B, C, D, None, delta_bias, delta_softplus
        )

        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.save_for_backward(x, x_dbl, x_proj_weight, delta_proj_weight, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """
        backward path
        """
        (x, x_dbl, x_proj_weight, delta_proj_weight, delta, 
         A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = x.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        x = x.to(dtype=torch.float32)
       
        if dout.stride(-1) != 1:
            dout = dout.contiguous().to(dtype=torch.float32)

        dx = torch.empty_like(x)


        d_x_out, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = selective_scan_cuda.bwd(
            x, delta, A, B, C, D, None, delta_bias, dout, scan_intermediates, out, None, ctx.delta_softplus, True
        )

        dD = dD.to(dtype=torch.float32) if D is not None else None

        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous().to(dtype=torch.float32)
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous().to(dtype=torch.float32)
            
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None

        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None

        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        
        

        ddelta_proj_weight = torch.zeros_like(torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank]))
        ddelta_proj_weight += torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])

        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dx_proj_weight = torch.zeros_like(torch.einsum("Br,Bd->rd", dx_dbl, rearrange(x, "b d l -> (b l) d")))
        dx_proj_weight += torch.einsum("Br,Bd->rd", dx_dbl, rearrange(x, "b d l -> (b l) d"))

        return (dx, dx_proj_weight, ddelta_proj_weight,
                dA, dB, dC, dD, ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


def mamba_inner_fn_no_out_proj(
    x, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFnNoOutProj.apply(x, x_proj_weight, delta_proj_weight,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)

'''
class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
     
        out, x = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus)

        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)

        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()


        du, ddelta, dA, dB, dC, dD, ddelta_bias = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, None, ctx.delta_softplus, False
        )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                ddelta_bias if delta_bias is not None else None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state)




class MambaInnerFnNoOutProj(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
        forward path - Conv1D 연산 제거, SSM 연산만 수행
        """
        assert checkpoint_lvl in [0, 1]
        L = x.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())

        if x.stride(-1) != 1:
            x = x.contiguous()

        x = x.to(dtype=torch.float32)

        # ✅ Linear projection (Conv1D 제거됨)
        x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        delta = delta.contiguous().to(dtype=torch.float32)

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None

        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous().to(dtype=torch.float32)

        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous().to(dtype=torch.float32)

        if D is not None:
            D = D.contiguous().to(dtype=torch.float32)

        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)
        C = C.to(dtype=torch.float32)
        D = D.to(dtype=torch.float32)


        out, scan_intermediates = selective_scan_cuda.fwd(
            x, delta, A, B, C, D, delta_bias, delta_softplus
        )

        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.save_for_backward(x, x_dbl, x_proj_weight, delta_proj_weight, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """
        backward path
        """
        (x, x_dbl, x_proj_weight, delta_proj_weight, delta, 
         A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = x.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        x = x.to(dtype=torch.float32)
       
        if dout.stride(-1) != 1:
            dout = dout.contiguous().to(dtype=torch.float32)

        dx = torch.empty_like(x)


        d_x_out, ddelta, dA, dB, dC, dD, ddelta_bias = selective_scan_cuda.bwd(
            x, delta, A, B, C, D, delta_bias, dout, scan_intermediates, out, ctx.delta_softplus, True
        )

        dD = dD.to(dtype=torch.float32) if D is not None else None

        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous().to(dtype=torch.float32)
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous().to(dtype=torch.float32)
            
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None

        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None

        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(x, "b d l -> (b l) d"))

        return (dx, dx_proj_weight, ddelta_proj_weight,
                dA, dB, dC, dD, ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


def mamba_inner_fn_no_out_proj(
    x, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFnNoOutProj.apply(x, x_proj_weight, delta_proj_weight,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)
'''