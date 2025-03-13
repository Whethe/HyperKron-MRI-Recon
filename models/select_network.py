'''
# -----------------------------------------
Define Training Network
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import functools

import numpy as np
import torch
from torch.nn import init


# --------------------------------------------
# Recon Generator, netG, G
# --------------------------------------------
def define_G(opt, **kwargs):
    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    # load from kwargs
    if 'load_netG1' in kwargs:
        opt['netG'] = opt['netG1']
    if 'load_netG2' in kwargs:
        opt['netG'] = opt['netG2']

    opt_net = opt['netG']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # SwinIR (for SwinMR)
    # ----------------------------------------
    if net_type == 'swinMR':
        if not opt_net['out_chans']:
            opt_net['out_chans'] = opt_net['in_chans']
        from models.network.network_swinmr import SwinIR as net
        netG = net(img_size=opt_net['img_size'],
                   in_chans=opt_net['in_chans'],
                   out_chans=opt_net['out_chans'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upscale=opt_net['upscale'],
                   img_range=opt_net['img_range'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],)

    elif net_type == 'swinMR_hyper':
        if not opt_net['out_chans']:
            opt_net['out_chans'] = opt_net['in_chans']
        from models.network.network_swinmr_hyper import SwinHyperIR as net
        assert opt_net['ph_n'] is not None
        print(opt_net['img_size'])
        netG = net(img_size=opt_net['img_size'],
                   in_chans=opt_net['in_chans'],
                   out_chans=opt_net['out_chans'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upscale=opt_net['upscale'],
                   img_range=opt_net['img_range'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],
                   ph_n=opt_net['ph_n'],)

    elif net_type == 'swinMR_hyper_n=4':
        if not opt_net['out_chans']:
            opt_net['out_chans'] = opt_net['in_chans']
        from models.network.network_swinmr_hyper_n4 import SwinHyperIR as net
        assert opt_net['ph_n'] is not None
        print(opt_net['img_size'])
        netG = net(img_size=opt_net['img_size'],
                   in_chans=opt_net['in_chans'],
                   out_chans=opt_net['out_chans'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upscale=opt_net['upscale'],
                   img_range=opt_net['img_range'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],)

    elif net_type == 'unet':
        from models.network.network_unet_v2 import UNet as net
        netG = net(
            T=1000,
            in_channels=opt_net['in_channels'],
            out_channels=opt_net['out_channels'],
            ch=opt_net['model_channels'],
            ch_mult=opt_net['channel_mult'],
            num_res_blocks=opt_net['num_res_blocks'],
            attn=opt_net['attention_resolutions'],
            dropout=opt_net['dropout'],
        )

    elif net_type == 'unet_hyper':
        from models.network.network_hyperunet import PHUNet as net
        netG = net(
            ph_n=opt_net['ph_n'],
            T=1000,
            in_channels=opt_net['in_channels'],
            out_channels=opt_net['out_channels'],
            ch=opt_net['model_channels'],
            ch_mult=opt_net['channel_mult'],
            num_res_blocks=opt_net['num_res_blocks'],
            attn=opt_net['attention_resolutions'],
            dropout=opt_net['dropout'],
        )


    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt, type='D'):
    if type == 'D':
        opt_net = opt['netD']
    elif type == 'D_g':
        opt_net = opt['netD_g']
    else:
        raise NotImplementedError(f'Unknown Discriminator type {type}')
    net_type = opt_net['net_type']


    # (Recommended) discriminator_unet
    # ----------------------------------------
    if net_type == 'discriminator_unet':
        from models.network.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])


    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


# --------------------------------------------
# weights initialization
# --------------------------------------------
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        # Excecption for ViGU
        EXCP_LIST = ['BasicConv',
                     'MRConv2d',
                     'EdgeConv2d',
                     'GINConv2d',
                     'GraphSAGE',
                     'DyGraphConv2d',
                     'DoubleConvBlock',
                     'TriConvBlockV2',
                     'ConvBlock']
        EXCP = np.prod([classname.find(exp) == -1 for exp in EXCP_LIST])

        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and EXCP:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')

        class DictToObject:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    setattr(self, key, value)
