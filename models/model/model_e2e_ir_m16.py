'''
# -----------------------------------------
Model
E2E IR m.1.6
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from models.select_network import define_G
from models.model.model_base import ModelBase
from models.loss import CharbonnierLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss

from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_fourier import *

import wandb
from math import ceil


class MRI_E2E_Recon(ModelBase):

    def __init__(self, opt):
        super(MRI_E2E_Recon, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.opt_dataset = self.opt['datasets']
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['freeze_patch_embedding']:
            for para in self.netG.module.patch_embed.parameters():
                para.requires_grad = False
            print("Patch Embedding Frozen (Requires Grad)!")
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        if opt['rank'] == 0:
            wandb.watch(self.netG)


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log


    # ----------------------------------------
    # load pre-trained G and E model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        # do not try to use 'use_input_norm' or 'use_range_norm' to support ch==2 input.
        # the pretrained weight only support ch=3 (or ch=1 by broadcast)input.
        self.perceptual_lossfn = PerceptualLoss().to(self.device)


    def total_loss(self):

        self.alpha = self.opt_train['alpha']
        self.beta = self.opt_train['beta']
        self.gamma = self.opt_train['gamma']

        # H HR, E Recon, L LR:
        self.H_2ch = self.H_SC.clone()
        self.E_2ch = self.E.clone()
        self.H_complex = torch.complex(self.H_2ch[:, 0:1, :, :], self.H_2ch[:, 1:, :, :])
        self.E_complex = torch.complex(self.E_2ch[:, 0:1, :, :], self.E_2ch[:, 1:, :, :])
        self.H_1ch = torch.abs(self.H_complex)
        self.E_1ch = torch.abs(self.E_complex)

        self.loss_image = self.G_lossfn(self.E_2ch, self.H_2ch)  # 2CH
        self.H_k_real, self.H_k_imag = fft_map(self.H_1ch)
        self.E_k_real, self.E_k_imag = fft_map(self.E_1ch)
        self.loss_freq = (self.G_lossfn(self.E_k_real, self.H_k_real) + self.G_lossfn(self.E_k_imag, self.H_k_imag)) / 2
        self.loss_perc = self.perceptual_lossfn(self.E_1ch, self.H_1ch)

        return self.alpha * self.loss_image + self.beta * self.loss_freq + self.gamma * self.loss_perc

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        if self.opt_train['G_optimizer_type'] == 'adam':
            if self.opt_train['freeze_patch_embedding']:
                self.G_optimizer = Adam(filter(lambda p: p.requires_grad, G_optim_params), lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'])
                print("Patch Embedding Frozen (Optimizer)!")
            else:
                self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'])
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            if self.opt_train['freeze_patch_embedding']:
                self.G_optimizer = AdamW(filter(lambda p: p.requires_grad, G_optim_params), lr=self.opt_train['G_optimizer_lr'],  weight_decay=self.opt_train['G_optimizer_wd'])
                print("Patch Embedding Frozen (Optimizer)!")
            else:
                self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.H_ABS = data['H_ABS'].to(self.device)
        self.L_ABS = data['L_ABS'].to(self.device)
        self.H_SC = data['H_SC'].to(self.device)
        self.L_SC = data['L_SC'].to(self.device)
        # self.mask = data['mask'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L_SC)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.current_step = current_step
        self.G_optimizer.zero_grad()
        self.netG_forward()

        G_loss = self.G_lossfn_weight * self.total_loss()
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # ------------------------------------
        # record log
        # ------------------------------------
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_image'] = self.loss_image.item()
        self.log_dict['G_loss_frequency'] = self.loss_freq.item()
        self.log_dict['G_loss_preceptual'] = self.loss_perc.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def record_loss_for_val(self):

        G_loss = self.G_lossfn_weight * self.total_loss()

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['G_loss_image'] = self.loss_image.item()
        self.log_dict['G_loss_frequency'] = self.loss_freq.item()
        self.log_dict['G_loss_preceptual'] = self.loss_perc.item()


    def check_windowsize(self):
        raise NotImplementedError
        self.window_size = self.opt['netG']['window_size']
        _, _, h_old, w_old = self.H.size()
        h_pad = ceil(h_old / self.window_size) * self.window_size - h_old  # downsampling for 3 times (2^3=8)
        w_pad = ceil(w_old / self.window_size) * self.window_size - w_old
        self.h_old = h_old
        self.w_old = w_old
        self.H = torch.cat([self.H, torch.flip(self.H, [2])], 2)[:, :, :h_old + h_pad, :]
        self.H = torch.cat([self.H, torch.flip(self.H, [3])], 3)[:, :, :, :w_old + w_pad]
        self.L = torch.cat([self.L, torch.flip(self.L, [2])], 2)[:, :, :h_old + h_pad, :]
        self.L = torch.cat([self.L, torch.flip(self.L, [3])], 3)[:, :, :, :w_old + w_pad]

    def recover_windowsize(self):
        raise NotImplementedError
        self.L = self.L[..., :self.h_old, :self.w_old]
        self.H = self.H[..., :self.h_old, :self.w_old]
        self.E = self.E[..., :self.h_old, :self.w_old]

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()


    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L_ABS.detach()[0].float().cpu()
        out_dict['E'] = self.E_1ch.detach()[0].float().cpu()
        out_dict['H'] = self.H_ABS.detach()[0].float().cpu()
        return out_dict

    def current_visuals_gpu(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L_ABS.detach()[0].float()
        out_dict['E'] = self.E_1ch.detach()[0].float()
        out_dict['H'] = self.H_ABS.detach()[0].float()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L_ABS.detach().float().cpu()
        out_dict['E'] = self.E_1ch.detach().float().cpu()
        out_dict['H'] = self.H_ABS.detach().float().cpu()
        return out_dict

    def current_results_gpu(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L_ABS.detach().float()
        out_dict['E'] = self.E_1ch.detach().float()
        out_dict['H'] = self.H_ABS.detach().float()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
