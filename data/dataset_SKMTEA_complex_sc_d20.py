'''
# -----------------------------------------
Data Loader
STK-TEA d.2.0.Complex.SC
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import random
import h5py
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_base import *
from models.select_mask import define_Mask
from math import floor
from skimage.transform import resize
from scipy.fftpack import *


def read_h5(data_path, is_segmentation=False):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        dict['image_complex'] = file['image_complex'][()]
        dict['data_name'] = file['image_complex'].attrs['data_name']
        dict['slice_idx'] = file['image_complex'].attrs['slice_idx']
        dict['segmenation_mask'] = file['segmenation_mask'][()] if is_segmentation else None

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


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
    hot encoding vector

    """

    _mask = [mask == i+1 for i in range(num_classes)]
    _mask = np.transpose(np.array(_mask), (1, 2, 0)).astype(int)

    return _mask

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """

    _mask = np.argmax(mask, axis=0).astype(int)
    return _mask


class DatasetSKMTEA(data.Dataset):

    def __init__(self, opt):
        super(DatasetSKMTEA, self).__init__()

        self.opt = opt
        self.patch_size = self.opt['H_size']
        self.is_mini_dataset = self.opt['is_mini_dataset']
        self.mini_dataset_prec = self.opt['mini_dataset_prec']

        self.is_augmentation = self.opt['is_augmentation']
        self.normalisation_type = self.opt['normalisation_type']
        self.is_segmentation = self.opt['is_segmentation'] if 'is_segmentation' in self.opt.keys() else False

        # get data path
        self.paths_raw = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []
        for path in self.paths_raw:
            if 'MTR' in path:
                self.paths_H.append(path)
            else:
                raise ValueError('Error: Unknown filename is in raw path')

        if self.is_mini_dataset:
            pass

        # get mask
        if 'fMRI' in self.opt['mask']:
            mask_1d = define_Mask(self.opt)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, 512, axis=1).transpose((1, 0))
            self.mask = mask  # (H, W)
        else:
            self.mask = define_Mask(self.opt)  # (H, W)

    def __getitem__(self, index):

        mask = self.mask  # H, W, 1

        # get gt image
        H_path = self.paths_H[index]

        img_dict = read_h5(H_path, is_segmentation=self.is_segmentation)

        img_H_SC = img_dict['image_complex']

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
        img_L_ABS = img_L_ABS[:, :, np.newaxis]  # H, W, 1
        img_H_ABS = img_H_ABS[:, :, np.newaxis]  # H, W, 1

        # Complex --> 2CH
        img_H_SC = np.concatenate((np.real(img_H_SC), np.imag(img_H_SC)), axis=-1)  # H, W, 2
        img_L_SC = np.concatenate((np.real(img_L_SC), np.imag(img_L_SC)), axis=-1)  # H, W, 2

        # get image information
        data_name = img_dict['data_name']
        slice_idx = img_dict['slice_idx']
        img_info = '{}_{:03d}'.format(data_name, slice_idx)

        # segmentation mask
        if self.is_segmentation:
            seg_mask = img_dict['segmenation_mask']
            seg_mask[seg_mask == 4] = 3
            seg_mask[seg_mask == 5] = 4
            seg_mask[seg_mask == 6] = 4
            seg_mask_one_hot = mask2onehot(seg_mask, num_classes=4)

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
            patch_L_ABS = img_L_ABS[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H_ABS = img_H_ABS[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            if self.is_augmentation:
                mode = random.randint(0, 7)
                patch_L_SC = util.augment_img(patch_L_SC, mode=mode)
                patch_H_SC = util.augment_img(patch_H_SC, mode=mode)
                patch_L_ABS = util.augment_img(patch_L_ABS, mode=mode)
                patch_H_ABS = util.augment_img(patch_H_ABS, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L_SC = torch.from_numpy(np.ascontiguousarray(patch_L_SC)).permute(2, 0, 1).to(torch.float32)
            img_H_SC = torch.from_numpy(np.ascontiguousarray(patch_H_SC)).permute(2, 0, 1).to(torch.float32)
            img_L_ABS = torch.from_numpy(np.ascontiguousarray(patch_L_ABS)).permute(2, 0, 1).to(torch.float32)
            img_H_ABS = torch.from_numpy(np.ascontiguousarray(patch_H_ABS)).permute(2, 0, 1).to(torch.float32)

            out_dict = {'L_SC': img_L_SC,  # (2, H, W)
                        'L_ABS': img_L_ABS,  # (1, H, W)
                        'H_SC': img_H_SC,  # (2, H, W)
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
            img_L_ABS = torch.from_numpy(np.ascontiguousarray(img_L_ABS)).permute(2, 0, 1).to(torch.float32)
            img_H_ABS = torch.from_numpy(np.ascontiguousarray(img_H_ABS)).permute(2, 0, 1).to(torch.float32)

            out_dict = {'L_SC': img_L_SC,  # (2, H, W)
                        'L_ABS': img_L_ABS,  # (1, H, W)
                        'H_SC': img_H_SC,  # (2, H, W)
                        'H_ABS': img_H_ABS,  # (1, H, W)
                        'H_path': H_path,
                        'mask': mask,  # (H, W)
                        'img_info': img_info,
                        'data_name': data_name,
                        'slice_idx': slice_idx,
                        'v_max': v_max,
                        'v_min': v_min}

            if self.is_segmentation:
                seg_mask_one_hot = torch.from_numpy(np.ascontiguousarray(seg_mask_one_hot)).permute(2, 0, 1)
                out_dict['seg_mask'] = seg_mask_one_hot

        return out_dict

    def __len__(self):
        return len(self.paths_H)






