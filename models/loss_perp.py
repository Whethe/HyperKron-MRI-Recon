import torch
import torch.nn as nn
import torch.nn.functional as F


class PerpLoss(nn.Module):
    """
    Perpendicular loss
    Reference: https://gitlab.com/computational-imaging-lab/perp_loss
    """

    def __init__(self):
        super().__init__()
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        # assert X.is_complex()
        # assert Y.is_complex()

        X = torch.complex(X[:, 0, ...], X[:, 1, ...])
        Y = torch.complex(Y[:, 0, ...], Y[:, 1, ...])

        mag_input = torch.abs(X)
        mag_target = torch.abs(Y)
        cross = torch.abs(X.real * Y.imag - X.imag * Y.real)

        angle = torch.atan2(X.imag, X.real) - torch.atan2(Y.imag, Y.real)
        ploss = torch.abs(cross) / (mag_input + 1e-8)

        aligned_mask = (torch.cos(angle) < 0).bool()

        final_term = torch.zeros_like(ploss)
        final_term[aligned_mask] = mag_target[aligned_mask] + (mag_target[aligned_mask] - ploss[aligned_mask])
        final_term[~aligned_mask] = ploss[~aligned_mask]
        return (final_term + F.mse_loss(mag_input, mag_target)).mean()