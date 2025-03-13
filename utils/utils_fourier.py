import torch
import torch.fft
from typing import List, Optional

"""
# --------------------------------------------
# Jiahao Huang (j.huang21@imperial.uk.ac)
# 30/Jan/2022
# --------------------------------------------
"""


# Fourier Transform
def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag

    return fft_x_real, fft_x_imag


def fft2_new(data, norm="ortho"):

    tmp = torch.fft.ifftshift(data, dim=(-2, -1))
    tmp = torch.fft.fftn(tmp, dim=(-2, -1), norm=norm)
    tmp = torch.fft.fftshift(tmp, dim=(-2, -1))

    return tmp


def fft2_complex_new(data, norm="ortho"):

    data_complex = torch.complex(data[..., :1, :, :], data[..., 1:, :, :])

    tmp = torch.fft.ifftshift(data_complex, dim=(-2, -1))
    tmp = torch.fft.fftn(tmp, dim=(-2, -1), norm=norm)
    tmp = torch.fft.fftshift(tmp, dim=(-2, -1))

    tmp = torch.concat((tmp.real, tmp.imag), dim=-3)

    return tmp

def ifft2_new(data, norm="ortho"):

    tmp = torch.fft.ifftshift(data, dim=(-2, -1))
    tmp = torch.fft.ifftn(tmp, dim=(-2, -1), norm=norm)
    tmp = torch.fft.fftshift(tmp, dim=(-2, -1))

    return tmp


def ifft2_complex_new(data, norm="ortho"):

    data_complex = torch.complex(data[..., :1, :, :], data[..., 1:, :, :])

    fft_tmp = torch.fft.ifftshift(data_complex, dim=(-2, -1))

    # FIXME: DELETE
    # check if there is nan in tmp
    if torch.isnan(fft_tmp).any():
        raise ValueError('nan before ifft')

    tmp = torch.fft.ifftn(fft_tmp, dim=(-2, -1), norm=norm)

    # FIXME: DELETE
    # check if there is nan in tmp
    if torch.isnan(tmp).any():
        raise ValueError('nan after ifft')

    tmp = torch.fft.fftshift(tmp, dim=(-2, -1))
    tmp = torch.concat((tmp.real, tmp.imag), dim=-3)

    return tmp


# --------------------------------------------
# from fastMRI Official repo
# --------------------------------------------

# # fft 2D for complex data
# def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2 dimensional Fast Fourier Transform.
#
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.fft``.
#
#     Returns:
#         The FFT of the input.
#     """
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")
#
#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.fftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])
#
#     return data
#
# # ifft 2D for complex data
# def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.ifft``.
#
#     Returns:
#         The IFFT of the input.
#     """
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")
#
#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.ifftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])
#
#     return data
#

# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

