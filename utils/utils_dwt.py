import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_wavelets import DWTForward, DWTInverse


def dwt2d_j1_tensor2tensor(x):
    # x (B, C, H, W)
    # x_dwt (B, C, 4, H, W)

    idwt_func = DWTForward(wave='haar', mode='zero').to(x.device)

    Yl, Yh = idwt_func(x)  # x (B, C, H, W), Yl (B, C, H, W), Yh [(B, C, 3, H, W), ]

    ll = Yl
    lh = Yh[0][:, :, 0, :, :]
    hl = Yh[0][:, :, 1, :, :]
    hh = Yh[0][:, :, 2, :, :]

    x_dwt = torch.stack((ll, lh, hl, hh), dim=2)

    return x_dwt  # x_dwt (B, C, 4, H, W)



def idwt2d_j1_tensor2tensor(x_dwt):
    # x_dwt (B, C, 4, H, W)
    # x_idwt (B, C, H, W)

    idwt_func = DWTInverse(wave='haar', mode='zero').to(x_dwt.device)

    ll = x_dwt[:, :, 0, :, :]
    lh = x_dwt[:, :, 1, :, :]
    hl = x_dwt[:, :, 2, :, :]
    hh = x_dwt[:, :, 3, :, :]

    Yl = ll
    _Yh = torch.stack((lh, hl, hh), dim=2)
    Yh = [_Yh]

    x_idwt = idwt_func((Yl, Yh))  # x_idwt (B, C, H, W), Yl (B, C, H, W), Yh [(B, C, 3, H, W), ]

    return x_idwt  # x_idwt (B, C, H, W)


if __name__ == '__main__':

    # Load and prepare the image
    img = Image.open("example/GT_00000.png").convert('L')  # Convert to grayscale
    img_tensor = ToTensor()(img).unsqueeze(0)  # Add batch dimension
    img_tensor = torch.concat([img_tensor, img_tensor], dim=1)  # B C H W
    img_tensor = torch.concat([img_tensor, img_tensor, img_tensor, img_tensor, img_tensor], dim=0)  # B C H W

    # img_tensor: 0~1 float tensor, shape (5, 2, H, W), 5: batch size, 2: channel, H: height, W: width

    # Perform 2D DWT
    img_dwt = dwt2d_j1_tensor2tensor(img_tensor)  # img_tensor (B, C, H, W) -> img_dwt (B, C, 4, H, W)

    # Perform 2D iDWT
    img_tensor_new = idwt2d_j1_tensor2tensor(img_dwt)  # img_dwt (B, C, 4, H, W) --> img_tensor_new (B, C, H, W)

    assert torch.allclose(img_tensor, img_tensor_new, atol=1e-6), "Error in DWT and IDWT"

    # save example
    # img_tensor = img_tensor[0, 0, :, :].numpy()
    # img_dwt_ll = img_dwt[0, 0, 0, :, :].numpy()
    # img_dwt_lh = img_dwt[0, 1, 0, :, :].numpy()
    # img_dwt_hl = img_dwt[0, 2, 0, :, :].numpy()
    # img_dwt_hh = img_dwt[0, 3, 0, :, :].numpy()
    # img_tensor_new = img_tensor_new[0, 0, :, :].numpy()
    # plt.imsave("tmp/GT_00000.png", img_tensor, cmap='gray')
    # plt.imsave("tmp/GT_00000_dwt_ll.png", img_dwt_ll, cmap='gray')
    # plt.imsave("tmp/GT_00000_dwt_lh.png", img_dwt_lh, cmap='gray')
    # plt.imsave("tmp/GT_00000_dwt_hl.png", img_dwt_hl, cmap='gray')
    # plt.imsave("tmp/GT_00000_dwt_hh.png", img_dwt_hh, cmap='gray')
    # plt.imsave("tmp/GT_00000_new.png", img_tensor_new, cmap='gray')

