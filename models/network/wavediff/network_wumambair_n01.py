# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

import functools
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import sys
from pathlib import Path
import copy

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
try:
    from .DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
    from . import dense_layer, layers, layerspp, utils
    from .mamba_ops import VSSBlock, selective_scan_flop_jit, parameter_count ,flop_count
except:
    from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
    import dense_layer, layers, layerspp, utils
    from mamba_ops import VSSBlock, selective_scan_flop_jit, parameter_count ,flop_count



ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn_update
WaveletResnetBlockBigGAN = layerspp.WaveletResnetBlockBigGANpp_Adagn
IdentityBlock = layerspp.IdentityBlock

Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@utils.register_model(name='wavelet_ncsnpp')
class WaveletNCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.patch_size = config.patch_size
        assert config.image_size % self.patch_size == 0

        self.ssm_d_state = ssm_d_state = config.ssm_d_state
        self.ssm_ratio = ssm_ratio = config.ssm_ratio
        self.ssm_rank_ratio = ssm_rank_ratio = config.ssm_rank_ratio
        self.ssm_dt_rank = ssm_dt_rank = config.ssm_dt_rank
        self.ssm_conv = ssm_conv = config.ssm_conv
        self.ssm_conv_bias = ssm_conv_bias = config.ssm_conv_bias
        self.ssm_drop_rate = ssm_drop_rate = config.ssm_drop_rate
        self.ssm_simple_init = ssm_simple_init = config.ssm_simple_init
        self.softmax_version = softmax_version = config.softmax_version
        self.forward_type = forward_type = config.forward_type
        self.mlp_ratio = mlp_ratio = config.mlp_ratio
        self.mlp_drop_rate = mlp_drop_rate = config.mlp_drop_rate
        self.use_checkpoint = use_checkpoint = config.use_checkpoint

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        resblock_dropout = config.resblock_dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            (config.image_size // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        # self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none']
        assert progressive_input in ['residual']
        # assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        self.no_use_fbn = getattr(self.config, "no_use_fbn", False)
        self.no_use_freq = getattr(self.config, "no_use_freq", False)
        self.no_use_residual = getattr(self.config, "no_use_residual", False)
        self.no_use_attn = getattr(self.config, "no_use_attn", False)

        modules = []

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        if self.no_use_residual:
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)
        else:
            pyramid_downsample = functools.partial(layerspp.WaveletDownsample)

        if resblock_type == 'mamba_vss_v2':
            if self.no_use_freq:
                ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)
            else:
                ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)

            VSSB = functools.partial(VSSBlock,
                                     dropout_path=0.0,
                                     norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
                                     ssm_d_state=ssm_d_state,
                                     ssm_ratio=ssm_ratio,
                                     ssm_rank_ratio=ssm_rank_ratio,
                                     ssm_dt_rank=ssm_dt_rank,
                                     ssm_act_layer=nn.SiLU,
                                     ssm_conv=ssm_conv,
                                     ssm_conv_bias=ssm_conv_bias,
                                     ssm_drop_rate=ssm_drop_rate,
                                     ssm_simple_init=ssm_simple_init,
                                     softmax_version=softmax_version,
                                     forward_type=forward_type,
                                     mlp_ratio=mlp_ratio,
                                     mlp_act_layer=nn.GELU,
                                     mlp_drop_rate=mlp_drop_rate,
                                     use_checkpoint=use_checkpoint,)

        elif resblock_type == 'biggan':
            if self.no_use_freq:
                ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)
            else:
                ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)
            VSSB = ResnetBlock

        elif resblock_type == 'none':  # for ablation study
            if self.no_use_freq:
                ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)
            else:
                ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=resblock_dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=None,
                                                zemb_dim=None)

            VSSB = functools.partial(IdentityBlock,)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        channels = config.num_channels * self.patch_size**2
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        hs_c2 = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(VSSB(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                hs_c2.append(in_ch)

                modules.append(ResnetBlock(down=True, in_ch=in_ch))

                modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        # Bottleneck
        in_ch = hs_c[-1]
        modules.append(VSSB(in_ch=in_ch, out_ch=in_ch))
        if not self.no_use_attn:
            modules.append(AttnBlock(channels=in_ch))
        modules.append(VSSB(in_ch=in_ch, out_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(VSSB(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                if self.no_use_freq:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))

        assert not hs_c

        channels = getattr(config, "num_out_channels", channels)
        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                    num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x):
        # Patchify
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w", p1=self.patch_size, p2=self.patch_size)  # (B, C, H, W) -> (B, C*p*p, H//p, W//p)

        # Set timestep/noise_level embedding; only for continuous training
        zemb = None
        temb = None

        # Load modules
        modules = self.all_modules
        m_idx = 0

        # Centering input data
        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling blocks
        input_pyramid = x

        hs = [modules[m_idx](x)]
        skipHs = []
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1])
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            # Downsample
            if i_level != self.num_resolutions - 1:
                if self.no_use_freq:
                    h = modules[m_idx](h, temb, zemb)
                else:
                    h, skipH = modules[m_idx](h, temb, zemb)
                    skipHs.append(skipH)
                m_idx += 1

                input_pyramid = modules[m_idx](input_pyramid)
                m_idx += 1
                if self.skip_rescale:
                    input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                else:
                    input_pyramid = input_pyramid + h
                h = input_pyramid

                hs.append(h)

        h = hs[-1]

        if self.no_use_fbn:
            h = modules[m_idx](h,)
        else:
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2.,)
            h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        # Attn block
        if not self.no_use_attn:
            h = modules[m_idx](h)
            m_idx += 1

        if self.no_use_fbn:
            h = modules[m_idx](h)
        else:
            # forward on original feature space
            h = modules[m_idx](h)
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2.)  # forward on wavelet space
            h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1))

                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                if self.no_use_freq:
                    h = modules[m_idx](h, temb, zemb)
                else:
                    h = modules[m_idx](h, temb, zemb, skipH=skipHs.pop())  # list.pop(): load the final one, and remove it from the list
                m_idx += 1

        assert not hs

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h

    def params_and_flops(self, shape):
        # shape = self.__input_shape__[1:]
        from flops_jit_ops import add_flop_jit, mul_flop_jit, div_flop_jit, gelu_flop_jit, softmax_flop_jit, tanh_flop_jit
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,

            "prim::PythonOp.DWTFunction_2D": None,
            "prim::PythonOp.IDWTFunction_2D": None,
            "aten::div": div_flop_jit,
            "aten::mul": mul_flop_jit,
            "aten::mul_": mul_flop_jit,
            "aten::add": add_flop_jit,
            "aten::gelu": gelu_flop_jit,
            "aten::softmax": softmax_flop_jit,
            "aten::tanh": tanh_flop_jit,
        }
        # supported_ops = None
        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        flops = sum(Gflops.values()) * 1e9
        del model, input
        return params, flops


if __name__ == '__main__':
    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = 96
    print('waveamm')
    config = {}
    config['patch_size'] = 1
    config['image_size'] = res
    config['num_channels'] = 1
    config['num_channels_dae'] = 128
    config['centered'] = True
    config['num_res_blocks'] = 2
    config['ch_mult'] = [1, 2, 2, 2]
    config['attn_resolutions'] = [16]

    # valid, but do not change settings
    config['resblock_dropout'] = 0.0
    config['resamp_with_conv'] = True
    config['fir'] = True
    config['fir_kernel'] = [1, 3, 3, 1]
    config['skip_rescale'] = True
    config['no_use_fbn'] = False
    config['no_use_freq'] = False
    config['no_use_attn'] = False
    config['not_use_tanh'] = False
    config['resblock_type'] = 'mamba_vss_v2'
    # config['resblock_type'] = 'none'
    config['progressive'] = 'none'
    config['progressive_input'] = 'residual'
    config['progressive_combine'] = 'sum'

    # invalid
    config['nz'] = None
    config['z_emb_dim'] = None
    config['conditional'] = None
    config['embedding_type'] = None

    # mamba
    config['ssm_d_state'] = 16
    config['ssm_ratio'] = 2.0
    config['ssm_rank_ratio'] = 2.0
    config['ssm_dt_rank'] = "auto"
    config['ssm_conv'] = 3
    config['ssm_conv_bias'] = True
    config['ssm_drop_rate'] = 0.0
    config['ssm_simple_init'] = False
    config['softmax_version'] = False
    config['forward_type'] = "v2_mask"
    # config['forward_type'] = "v2_mask_identity"
    config['mlp_ratio'] = 2.0
    # config['mlp_ratio'] = 0.0
    config['mlp_drop_rate'] = 0.0
    config['use_checkpoint'] = False

    config = DictToObject(config)
    model = WaveletNCSNpp(config).to(device)

    batch = 1
    channel = 1
    height = 96
    width = res

    x = torch.randn(batch, channel, height, width).to(device)

    # with torch.no_grad():
    #     out = model(x)
    # print(x.shape)
    # print(out.shape)
    # print(model)

    with torch.no_grad():
        params, flops = model.params_and_flops((channel, height, width))
    print('Params: {:.3f}M'.format(params * 1e-6))
    print('FLOPs: {:.3f}G'.format(flops * 1e-9))


    # import time
    # with torch.no_grad():
    #     t_list = []
    #     for i in range(50):
    #         t1 = time.time()
    #         _ = model(x)
    #         t2 = time.time()
    #         t_used = t2 - t1
    #         t_list.append(t_used)
    # t_list = t_list[-10:]
    #
    # t_avg = np.average(t_list)
    # t_std = np.std(t_list)
    #
    # print('Time used: {}s'.format(round(t_avg, 3)))
    # print('Time used: {}s'.format(round(t_std, 3)))