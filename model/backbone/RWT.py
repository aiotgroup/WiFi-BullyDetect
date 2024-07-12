import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, LayerScale, Attention
import numpy as np

from ..function import (
    WaveAttention_res
)

from ..model_config import ModelConfig

import logging
logger = logging.getLogger( )

class RWT_config(ModelConfig):
    num_classes     = 7
    seq_len         = 224
    n_channel       = 52
    dim_mlp_hidden  = 2048
    expansion_factor = 4
    pooling         = False
    MAX_PATCH_NUMS  = 1000

    d_model         = 768
    num_layer       = 12
    n_head          = 12

    norm_layer  = nn.LayerNorm

    def __init__(self, config):
        super(RWT_config, self).__init__(config["model"]["backbone_setting"]["backbone_name"])

        self.attn_type          = config["model"]["backbone_setting"]["attn_type"]
        self.attn_type_layer    = config["model"]["backbone_setting"]["attn_type_layer"]
        self.scale              = config["model"]["backbone_setting"]["scale"]
        self.patch_size         = config["model"]["backbone_setting"]["patch_size"]
        self.dropout            = config["model"]["backbone_setting"]["dropout"]
        self.droppath           = config["model"]["backbone_setting"]["droppath"]
        self.high_ratio         = config["model"]["backbone_setting"]["others"]["high_ratio"]

        assert isinstance(self.attn_type_layer, int)
        assert isinstance(self.patch_size, int)
        assert isinstance(self.dropout, float)
        assert isinstance(self.droppath, float)
        assert isinstance(self.high_ratio, float)

        if self.scale == 'es':
            # Extra Small
            self.d_model = 128
            self.num_layer = 2
            self.n_head = 4
        elif self.scale == 'ms':
            # Medium Small
            self.d_model = 256
            self.num_layer = 4
            self.n_head = 8
        elif self.scale == 's':
            # Small
            self.d_model = 512
            self.num_layer = 8
            self.n_head = 8
        elif self.scale == 'b':
            # Base
            self.d_model = 768
            self.num_layer = 12
            self.n_head = 12
        elif self.scale == 'test':
            # Base
            self.d_model = 768
            self.num_layer = 1
            self.n_head = 8

def RWT(config: RWT_config):

    model = WaveVit(
        dropout=config.dropout,
        drop_path=config.droppath,
        high_ratio=config.high_ratio,
        model_name=config.model_name,
        num_classes=config.num_classes,
        n_channel=config.n_channel,
        seq_len=config.seq_len,
        patch_size=config.patch_size,
        embed_dim=config.d_model,
        num_head=config.n_head,
        expansion_factor=config.expansion_factor,
        depth=config.num_layer,
        MAX_PATCH_NUMS=config.MAX_PATCH_NUMS,
        pooling=config.pooling,
        attn_type=config.attn_type,
        attn_type_layer=config.attn_type_layer,
        norm_layer=config.norm_layer
    )
    return model

class SimpleSpanCLSHead(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SimpleSpanCLSHead, self).__init__()

        self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        return self.head_2(self.head_1(features))

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 N_dim,
                 dim_mlp_hidden = 2048,
                 dropout = 0.5,
                 norm_layer = nn.LayerNorm,
                 attn_type = 'wave',
                 act_layer = nn.GELU,
                 qkv_bias = True,
                 attn_drop = 0.5,
                 init_values=None,
                 drop_path=0.1,
                 sr_ratio=1,
                 high_ratio=1.0
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=dropout)

        # --------------------------------------------------------------------------
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # --------------------------------------------------------------------------

        if attn_type == 'timm':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'waveres':
            self.attn = WaveAttention_res(dim=dim, N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=attn_drop, high_ratio=high_ratio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class WaveVit(nn.Module):
    def __init__(self,
                 model_name = 'wavevit_timm_s_16',
                 num_classes = 10,
                 n_channel = 90,
                 seq_len = 2000,
                 patch_size = 16,
                 embed_dim = 768,
                 num_head = 12,
                 # dim_mlp_hidden = 2048,
                 dropout = 0.5,
                 drop_path = 0.1,
                 high_ratio=1.0,
                 expansion_factor=4,
                 depth = 12,
                 MAX_PATCH_NUMS = 1000,
                 pooling = False,
                 attn_type='wave',
                 attn_type_layer = 4,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.seq_len = seq_len

        self.N_dim = (seq_len // patch_size) if (seq_len % patch_size == 0) else (seq_len // patch_size + 1)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.depth = depth
        self.MAX_PATCH_NUMS = MAX_PATCH_NUMS
        self.pooling = pooling
        self.dim_mlp_hidden = embed_dim * expansion_factor
        self.dropout = dropout
        # --------------------------------------------------------------------------
        #
        self.cls_embed = nn.Parameter(torch.empty((1, 1, embed_dim)), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.empty((1, 1 + MAX_PATCH_NUMS, embed_dim)),
                                      requires_grad=True)

        self.embedding = nn.Linear(n_channel * patch_size, embed_dim, bias=False)

        model_list = []
        for i in range(depth):
            self.attn_type = 'timm' if i >= attn_type_layer else attn_type
            model_list.append(Block(dim=embed_dim,
                                    num_heads=num_head,
                                    N_dim=self.N_dim,
                                    dim_mlp_hidden=self.dim_mlp_hidden,
                                    dropout=dropout,
                                    attn_drop=dropout,
                                    # attn_type='timm',
                                    drop_path=drop_path,
                                    attn_type=self.attn_type,
                                    sr_ratio=1,
                                    high_ratio=high_ratio))

        self.blocks = nn.ModuleList(model_list)
        self.norm = norm_layer(embed_dim)

        # self.head = SimpleSpanCLSHead(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.cls_embed.data)
        nn.init.xavier_uniform_(self.pos_embed.data)

        self.apply(self._init_weights)
        # --------------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pickup_patching(self, batch_data):
        # batch_size, n_channels, seq_len
        batch_size, n_channels, seq_len = batch_data.size()
        if (seq_len % self.patch_size != 0):
            batch_data = F.pad(batch_data,
                               (0, self.patch_size - (seq_len % self.patch_size)),
                               'constant', 0)
            batch_size, n_channels, seq_len = batch_data.size()
        assert seq_len % self.patch_size == 0

        batch_data = batch_data.view(batch_size, n_channels, seq_len // self.patch_size, self.patch_size)
        batch_data = batch_data.permute(0, 2, 1, 3)
        batch_data = batch_data.reshape(batch_size, seq_len // self.patch_size, n_channels * self.patch_size)
        return batch_data

    def forward(self, x):
        x = self._pickup_patching(x)
        x = self.embedding(x)
        batch_size, num_patches, _ = x.size()
        # 拼接CLS向量
        x = torch.cat((self.cls_embed.repeat(batch_size, 1, 1), x), dim=1)
        # 加上Position Embedding
        x = x + self.pos_embed.repeat(batch_size, 1, 1)[:, :1 + num_patches, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.pooling:
            x = torch.mean(x, dim=1)
        else:
            x = x[:, 0, :]
        # x = self.head(x)
        return x

    def get_output_size(self):
        return self.embed_dim

    def get_model_name(self):
        return self.model_name