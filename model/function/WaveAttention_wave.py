import torch
import torch.nn as nn
from .torch_wavelets_1D import DWT_1D, IDWT_1D
from timm.models.vision_transformer import _cfg, LayerScale, Attention

import logging
logger = logging.getLogger( )

class WaveAttention_res(nn.Module):
    def __init__(self,
                 dim,
                 N_dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.5,
                 proj_drop=0.5,
                 high_ratio=1.0
                 ):
        super().__init__()
        assert dim % 2 == 0, "dim should be divisible by 2"
        assert dim % num_heads == 0, 'dim should be divisible by num_heads // 2'
        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')
        self.attn_l = Attention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        self.high_ratio = high_ratio
        # self.attn_h = Attention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)

    def forward(self, x):
        # logging.debug(f'wave_res: {self.high_ratio}')

        # [128, 126, 768]   [B, N, D]
        B, N, _ = x.shape
        x = self.dwt(x)             # [128, 252, 384]   [B, 2*N, D//2]
        x_h = x[:,N:,:] * self.high_ratio
        x = x[:,:N,:]

        x = self.attn_l(x)
        # x_h = self.attn_h(x_h)

        x = torch.cat([x,x_h], dim=1)
        x = self.idwt(x)
        return x

if __name__ == '__main__':

    import numpy as np
    # def _pickup_patching(batch_data):
    #     # batch_size, n_channels, seq_len
    #     batch_size, n_channels, seq_len = batch_data.size()
    #     patch_size = 16
    #     assert seq_len % patch_size == 0
    #     batch_data = batch_data.view(batch_size, n_channels, seq_len // patch_size, patch_size)
    #     batch_data = batch_data.permute(0, 2, 1, 3)
    #     batch_data = batch_data.reshape(batch_size, seq_len // patch_size, n_channels * patch_size)
    #     return batch_data
    # inputs = np.ones((2, 30, 1600))
    # inputs = torch.from_numpy(inputs).float().to(torch.device('cuda'))
    # print(inputs.shape)
    # inputs = _pickup_patching(inputs)
    # print(inputs.shape)
    # wave_attn = WaveAttention2(dim=480).to(torch.device('cuda'))
    # wave_attn(inputs)