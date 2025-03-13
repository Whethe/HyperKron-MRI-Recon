
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops
import numpy as np


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, num_windows, window_size*window_size, dim]
        B_, N, HW, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each is [batch_size, num_heads, num_windows, window_size*window_size, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        out = (attn @ v).transpose(2, 3).reshape(B_, N, -1, C)
        out = self.proj(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, window_size*window_size, dim]
        B_, HW, C = x.shape

        qkv = self.qkv(x).reshape(B_, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Split into separate tensors

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, -1, C)
        out = self.proj(out)
        return out


class SwinMSABlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim = dim
        self.window_attn = WindowAttention(dim, window_size, num_heads)

    def partition_windows(self, x, window_size):
        """
        Partition the input tensor into windows.
        x shape: [batch_size, height, width, channels]
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def merge_windows(self, windows, window_size, H, W):
        """
        Merge the windowed tensor back into the original shape.
        windows shape: [num_windows * batch_size, window_size, window_size, channels]
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        # x shape: [batch_size, height, width, channels]
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, "Height and Width must be divisible by window size."

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = self.partition_windows(shifted_x, self.window_size)  # [num_windows * batch_size, window_size, window_size, channels]

        # Window attention
        attn_windows = self.window_attn(x_windows)  # Same shape as x_windows

        # Merge windows
        shifted_back_x = self.merge_windows(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_back_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_back_x

        return x


class MSABlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.attn = Attention(dim, num_heads)

    def forward(self, x):
        # x shape: [batch_size, height, width, channels]
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # [batch_size, height*width, channels]

        # attention
        x = self.attn(x)

        x = x.reshape(B, H, W, C)  # [batch_size, height, width, channels]

        return x


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.gelu = nn.GELU()
    def forward(self, x):
        identity = x

        x = x.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
        out = self.conv(x)
        out = self.gelu(out)
        out = out.permute(0, 2, 3, 1)

        return out + identity


if __name__ == "__main__":

    # Using the same parameters for demonstration
    window_size = 8
    dim = 128
    num_heads = 8
    batch_size = 8
    height = 64
    width = 64

    input = torch.rand(batch_size, height, width, dim)
    print(input.shape)  # Expected shape: [batch_size, height*width, dim]

    # Example usage without cyclic shift
    swin_block = SwinMSABlock(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=0)

    # Example usage with cyclic shift
    swin_block_with_shift = SwinMSABlock(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=window_size//2)

    output = swin_block(input)
    print(output.shape)  # Expected shape: [batch_size, height*width, dim]

    output = swin_block_with_shift(input)
    print(output.shape)  # Expected shape: [batch_size, height*width, dim]

    # Example usage of TransformerBlock
    transformer_block = MSABlock(dim=dim, num_heads=num_heads)
    output = transformer_block(input)
    print(output.shape)  # Expected shape: [batch_size, height*width, dim]

    # Example usage of ConvBlock
    cnn_block = ConvBlock(dim=dim)
    output = cnn_block(input)
    print(output.shape)  # Expected shape: [batch_size, height, width, dim]