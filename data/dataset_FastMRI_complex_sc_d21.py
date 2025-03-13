"""
# -----------------------------------------
Data Loader
FastMRI d.2.1.Complex.SC
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
"""
import argparse
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data
from scipy.fftpack import *

import utils.utils_image as util
from data.select_dataset import define_Dataset
from models.select_mask import define_Mask


def read_h5(data_path):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        dict['image_complex'] = file['image_complex'][()]
        dict['data_name'] = file['image_complex'].attrs['data_name']
        dict['slice_idx'] = file['image_complex'].attrs['slice_idx']
        dict['image_rss'] = file['image_rss'][()]

    return dict


def preprocess_normalisation(img, type='complex'):

    if type == 'complex_mag':
        img = img / np.abs(img).max()
    elif type == 'complex':
        """ normalizes the magnitude of complex-valued image to range [0, 1] """
        abs_img = normalize(np.abs(img))
        ang_img = normalize(np.angle(img))  # from scoreMRI paper
        img = abs_img * np.exp(1j * ang_img)
    elif type == '0_1':
        img = normalize(img)
    else:
        raise NotImplementedError

    return img


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def undersample_kspace(x, mask):

    x = ifftshift(x, axes=(-2, -1))
    fft = fftn(x, axes=(-2, -1))
    fft = fftshift(fft, axes=(-2, -1))

    fft = fft * mask

    fft = ifftshift(fft, axes=(-2, -1))
    x = ifftn(fft, axes=(-2, -1))
    x = fftshift(x, axes=(-2, -1))

    return x


class DatasetFastMRI(data.Dataset):

    def __init__(self, opt):
        super(DatasetFastMRI, self).__init__()

        self.opt = opt
        self.patch_size = self.opt['H_size']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']

        self.is_augmentation = self.opt['is_augmentation']
        self.normalisation_type = self.opt['normalisation_type']

        # get data path
        self.paths_raw = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []
        for path in self.paths_raw:
            if 'file' in path:
                self.paths_H.append(path)
            else:
                raise ValueError('Error: Unknown filename is in raw path')

        if self.is_mini_dataset:
            self.paths_H = self.paths_H[:int(self.mini_dataset_prec * (len(self.paths_H)))]

        # get mask
        if 'fMRI' in self.opt['mask']:
            mask_1d = define_Mask(self.opt)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, 320, axis=1).transpose((1, 0))
            self.mask = mask  # (H, W)
        else:
            self.mask = define_Mask(self.opt)  # (H, W)

    def __getitem__(self, index):

        mask = self.mask  # H, W, 1

        # get gt image
        H_path = self.paths_H[index]

        img_dict = read_h5(H_path)

        img_H_RSS = img_dict['image_rss']
        img_H_SC = img_dict['image_complex']

        img_H_RSS = preprocess_normalisation(img_H_RSS, type='0_1')
        v_max = np.max(np.abs(img_H_SC))
        v_min = 0
        img_H_SC = preprocess_normalisation(img_H_SC, type=self.normalisation_type)

        # get zf image
        img_L_SC = undersample_kspace(img_H_SC, mask)
        img_L_ABS = abs(img_L_SC)
        img_H_ABS = abs(img_H_SC)

        # expand dim
        img_H_SC = img_H_SC[:, :, np.newaxis]  # H, W, 1
        img_L_SC = img_L_SC[:, :, np.newaxis]  # H, W, 1
        img_H_RSS = img_H_RSS[:, :, np.newaxis]  # H, W, 1
        img_L_ABS = img_L_ABS[:, :, np.newaxis]  # H, W, 1
        img_H_ABS = img_H_ABS[:, :, np.newaxis]  # H, W, 1

        # # 提取实部和虚部
        # img_real = np.real(img_H_SC)
        # img_imag = np.imag(img_H_SC)
        #
        # img_real_L = np.real(img_L_SC)
        # img_imag_L = np.imag(img_L_SC)

        # # 如果数值范围不在 0~255 内，可以归一化到这个范围，并转换为 uint8 类型
        # img_real_norm = cv2.normalize(img_real, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # img_imag_norm = cv2.normalize(img_imag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #
        # img_real_norm_L = cv2.normalize(img_real_L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # img_imag_norm_L = cv2.normalize(img_imag_L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #
        # img_norm_L = cv2.normalize(img_L_ABS, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #
        # # 分别保存实部和虚部的图片
        # cv2.imwrite('img_H_SC_real.png', img_real_norm)
        # cv2.imwrite('img_H_SC_imag.png', img_imag_norm)
        #
        # cv2.imwrite('img_L_SC_real.png', img_real_norm_L)
        # cv2.imwrite('img_L_SC_imag.png', img_imag_norm_L)
        #
        # cv2.imwrite('img_L.png', img_norm_L)

        # Complex --> 2CH
        img_H_SC = np.concatenate((np.real(img_H_SC), np.imag(img_H_SC)), axis=-1)  # H, W, 2
        img_L_SC = np.concatenate((np.real(img_L_SC), np.imag(img_L_SC)), axis=-1)  # H, W, 2

        # get image information
        data_name = img_dict['data_name']
        slice_idx = img_dict['slice_idx']
        img_info = '{}_{:03d}'.format(data_name, slice_idx)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H_SC.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            patch_L_SC = img_L_SC[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :, ]
            patch_H_SC = img_H_SC[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :, ]
            patch_H_RSS = img_H_RSS[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L_ABS = img_L_ABS[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H_ABS = img_H_ABS[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            if self.is_augmentation:
                mode = random.randint(0, 7)
                patch_L_SC = util.augment_img(patch_L_SC, mode=mode)
                patch_H_SC = util.augment_img(patch_H_SC, mode=mode)
                patch_H_RSS = util.augment_img(patch_H_RSS, mode=mode)
                patch_L_ABS = util.augment_img(patch_L_ABS, mode=mode)
                patch_H_ABS = util.augment_img(patch_H_ABS, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L_SC = torch.from_numpy(np.ascontiguousarray(patch_L_SC)).permute(2, 0, 1).to(torch.float32)
            img_H_SC = torch.from_numpy(np.ascontiguousarray(patch_H_SC)).permute(2, 0, 1).to(torch.float32)
            img_H_RSS = torch.from_numpy(np.ascontiguousarray(patch_H_RSS)).permute(2, 0, 1).to(torch.float32)
            img_L_ABS = torch.from_numpy(np.ascontiguousarray(patch_L_ABS)).permute(2, 0, 1).to(torch.float32)
            img_H_ABS = torch.from_numpy(np.ascontiguousarray(patch_H_ABS)).permute(2, 0, 1).to(torch.float32)

            out_dict = {'L_SC': img_L_SC,  # (2, H, W)
                        'L_ABS': img_L_ABS,  # (1, H, W)
                        'H_SC': img_H_SC,  # (2, H, W)
                        'H_RSS': img_H_RSS,  # (1, H, W)
                        'H_ABS': img_H_ABS,  # (1, H, W)
                        'H_path': H_path,
                        'mask': mask,  # (H, W)
                        'img_info': img_info,
                        'data_name': data_name,
                        'slice_idx': slice_idx,
                        'v_max': v_max,
                        'v_min': v_min}

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L_SC = torch.from_numpy(np.ascontiguousarray(img_L_SC)).permute(2, 0, 1).to(torch.float32)
            img_H_SC = torch.from_numpy(np.ascontiguousarray(img_H_SC)).permute(2, 0, 1).to(torch.float32)
            img_H_RSS = torch.from_numpy(np.ascontiguousarray(img_H_RSS)).permute(2, 0, 1).to(torch.float32)
            img_L_ABS = torch.from_numpy(np.ascontiguousarray(img_L_ABS)).permute(2, 0, 1).to(torch.float32)
            img_H_ABS = torch.from_numpy(np.ascontiguousarray(img_H_ABS)).permute(2, 0, 1).to(torch.float32)

            out_dict = {'L_SC': img_L_SC,  # (2, H, W)
                        'L_ABS': img_L_ABS,  # (1, H, W)
                        'H_SC': img_H_SC,  # (2, H, W)
                        'H_RSS': img_H_RSS,  # (1, H, W)
                        'H_ABS': img_H_ABS,  # (1, H, W)
                        'H_path': H_path,
                        'mask': mask,  # (H, W)
                        'img_info': img_info,
                        'data_name': data_name,
                        'slice_idx': slice_idx,
                        'v_max': v_max,
                        'v_min': v_min}

        return out_dict

    def __len__(self):
        return len(self.paths_H)



if __name__ == '__main__':
    opt = "../options/UNET/train_unet_v2_FastMRIKneePD_fMRIRanAF8CF0.04PE320_ps2_res320_m.1.6.n.0.1.v2m.d.2.1.cplx.sc.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=opt, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=None)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--dist', default=False)

    from utils import utils_option as option
    opt = option.parse(parser.parse_args().opt, is_train=True)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)