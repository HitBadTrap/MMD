# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch.nn as nn
import torch
from .mlp import BaseMLP
from .mamba import MotionMamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from einops import rearrange

class DownBlock(nn.Module):
    def __init__(self, depth=4,
                 seqlen=196,
                 input_dim=512, 
                 output_dim=512,
                 embed_dim=512,
                 ssm_cfg=None, 
                 drop_path_rate=0,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 if_channel=False,
                 w_embed=True,
                 **kwargs):
        super(DownBlock, self).__init__()
        self.down = nn.MaxPool1d(2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            RMSNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.mamba = MotionMamba(depth=depth,
                 seqlen=seqlen,
                 latent_dim=output_dim,
                 embed_dim=embed_dim, 
                 ssm_cfg=ssm_cfg, 
                 drop_path_rate=drop_path_rate,
                 norm_epsilon = norm_epsilon, 
                 rms_norm = rms_norm, 
                 fused_add_norm = fused_add_norm,
                 residual_in_fp32=residual_in_fp32,
                 device=device,
                 dtype=dtype,
                 if_bidirectional=if_bidirectional,
                 if_channel=if_channel,
                 w_embed=w_embed,
                 **kwargs)

    def forward(self, motion_input, embed=None):
        motion_feats = self.down(rearrange(motion_input, "b l c-> b c l"))
        motion_feats = self.mlp(rearrange(motion_feats, "b c l-> b l c"))
        motion_feats = self.mamba(motion_feats, embed)

        return motion_feats
    
class UpBlock(nn.Module):
    def __init__(self, depth=4,
                 seqlen=196,
                 input_dim=512,
                 skip_dim=512,
                 output_dim=512, 
                 embed_dim=512,
                 ssm_cfg=None, 
                 drop_path_rate=0,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 if_channel=False,
                 w_embed=True,
                 **kwargs):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose1d(input_dim,input_dim,2,2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+skip_dim, output_dim),
            RMSNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.mamba = MotionMamba(depth=depth,
                 seqlen=seqlen,
                 latent_dim=output_dim,
                 embed_dim=embed_dim, 
                 ssm_cfg=ssm_cfg, 
                 drop_path_rate=drop_path_rate,
                 norm_epsilon = norm_epsilon, 
                 rms_norm = rms_norm, 
                 fused_add_norm = fused_add_norm,
                 residual_in_fp32=residual_in_fp32,
                 device=device,
                 dtype=dtype,
                 if_bidirectional=if_bidirectional,
                 if_channel=if_channel,
                 w_embed=w_embed,
                 **kwargs)

    def forward(self, motion_input, skip_feats, embed=None):
        motion_feats = self.up(rearrange(motion_input, "b l c-> b c l"))
        x = torch.cat([motion_feats, rearrange(skip_feats, "b l c-> b c l")], dim=1)
        motion_feats = self.mlp(rearrange(x, "b c l-> b l c"))
        motion_feats = self.mamba(motion_feats, embed)

        return motion_feats
    
class DiffMotionUNet(nn.Module):
    def __init__(self, latent_dim=512, 
                 seq=196, 
                 mlp_layers=1,
                 depth=4,
                 ssm_cfg=None, 
                 drop_path_rate=0,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = True, 
                 fused_add_norm=True,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 if_channel=False,
                 w_embed=True,
                 **kwargs):

        super().__init__()
        self.motion_mlp = BaseMLP(dim=latent_dim, seq=seq, num_layers=mlp_layers)
        self.mamba = MotionMamba(depth=8,
                 seqlen=seq,
                 latent_dim=latent_dim,
                 embed_dim=latent_dim, 
                 ssm_cfg=ssm_cfg, 
                 drop_path_rate=drop_path_rate,
                 norm_epsilon = norm_epsilon, 
                 rms_norm = rms_norm, 
                 fused_add_norm = fused_add_norm,
                 residual_in_fp32=residual_in_fp32,
                 device=device,
                 dtype=dtype,
                 if_bidirectional=if_bidirectional,
                 if_channel=if_channel,
                 w_embed=w_embed,
                 **kwargs)
        self.down1 = DownBlock(depth=4, seqlen=seq//2, embed_dim=latent_dim, input_dim=latent_dim, output_dim=latent_dim*2, if_bidirectional=if_bidirectional, if_channel=if_channel)
        self.down2 = DownBlock(depth=4, seqlen=seq//4, embed_dim=latent_dim, input_dim=latent_dim*2, output_dim=latent_dim*4, if_bidirectional=if_bidirectional, if_channel=if_channel)
        self.up2 = UpBlock(depth=4, seqlen=seq//2, embed_dim=latent_dim, input_dim=latent_dim*4, skip_dim=latent_dim*2, output_dim=latent_dim*2, if_bidirectional=if_bidirectional, if_channel=if_channel)
        self.up1 = UpBlock(depth=8, seqlen=seq, embed_dim=latent_dim, input_dim=latent_dim*2, skip_dim=latent_dim, output_dim=latent_dim, if_bidirectional=if_bidirectional, if_channel=if_channel)
        
    def forward(self, x, embed):
        x = self.motion_mlp([x, embed])[0]
        x1 = self.mamba(x,embed)
        x2 = self.down1(x1,embed)
        x3 = self.down2(x2,embed)
        x = self.up2(x3,x2,embed)
        x = self.up1(x,x1,embed)

        return x
