import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, dec_hi, dec_lo):
        x = x.contiguous()
        ctx.save_for_backward(dec_hi, dec_lo)
        # B, C, L
        ctx.shape = x.shape

        dim = x.shape[1]
        low = torch.nn.functional.conv1d(x, dec_lo.expand(dim, -1, -1), stride=2, groups=dim)
        hi = torch.nn.functional.conv1d(x, dec_hi.expand(dim, -1, -1), stride=2, groups=dim)
        x = torch.cat([low, hi], dim=1)
        # B, 2C, L
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            dec_hi, dec_lo = ctx.saved_tensors
            B, C, L = ctx.shape
            dx = dx.view(B, 2, -1, L//2)

            dx = dx.transpose(1,2).reshape(B, -1, L//2)
            filters = torch.cat([dec_lo, dec_hi], dim=0).repeat(C, 1, 1)
            dx = torch.nn.functional.conv_transpose1d(dx, filters, stride=2, groups=C)

        return dx, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        # B, 2C, L
        ctx.shape = x.shape

        B, _, L = x.shape
        x = x.view(B, 2, -1, L).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, L)
        filters = filters.repeat(C, 1, 1)
        x = torch.nn.functional.conv_transpose1d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters, = ctx.saved_tensors
            # filters = filters[0]
            B, C, L = ctx.shape
            # 这里的C是2C
            C = C // 2
            dx = dx.contiguous()

            rec_lo, rec_hi = torch.unbind(filters, dim=0)
            low = torch.nn.functional.conv1d(dx, rec_lo.unsqueeze(1).expand(C, -1, -1), stride = 2, groups = C)
            hi = torch.nn.functional.conv1d(dx, rec_hi.unsqueeze(1).expand(C, -1, -1), stride = 2, groups = C)
            dx = torch.cat([low, hi], dim=1)

        return dx, None

class IDWT_1D(nn.Module):
    def __init__(self, wave):
        super(IDWT_1D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        rec_hi = rec_hi.unsqueeze(0).unsqueeze(1)
        rec_lo = rec_lo.unsqueeze(0).unsqueeze(1)
        # 先是低频后是高频
        filters = torch.cat([rec_lo, rec_hi], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_1D(nn.Module):
    def __init__(self, wave):
        super(DWT_1D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        self.register_buffer('dec_hi', dec_hi.unsqueeze(0).unsqueeze(0))
        self.register_buffer('dec_lo', dec_lo.unsqueeze(0).unsqueeze(0))

        # self.dec_hi = self.dec_hi.to(dtype=torch.float16)
        # self.dec_lo = self.dec_lo.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.dec_hi, self.dec_lo)


if __name__ == '__main__':
    import numpy as np
    import torch

    inputs = np.ones((2, 3, 8))
    inputs = torch.from_numpy(inputs).to(dtype=torch.float16).to(torch.device('cuda'))
    dwt1d = DWT_1D(wave='haar').to(torch.device('cuda'))
    out = dwt1d(inputs)
    # print(out.shape)
    idwt1d = IDWT_1D(wave='haar').to(torch.device('cuda'))
    out_idwt = idwt1d(out)

    # print(inputs)
    # print(out)
    # print(out_idwt)