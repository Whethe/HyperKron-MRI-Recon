import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        num_out_ch (int): Channel number of output features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

    def params(self):
        params = self.num_feat * 3 * 9
        return params


class SPFUnit(nn.Module):
    def __init__(self, in_channels, out_channels, resize_type='none'):
        super(SPFUnit, self).__init__()

        self.resize_type = resize_type

        # Convolution Block x 2
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

        # Semantic Selection Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(out_channels // 2, out_channels)
        self.sigmoid = nn.Sigmoid()

        # Resize operations
        if resize_type == 'upsample':
            self.upsample = UpsampleOneStep(scale=2, num_feat=out_channels, num_out_ch=out_channels)
        elif resize_type == 'downsample':
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif resize_type == 'none':
            self.identity_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, Fi, F_spf_i_minus_1):
        # Concatenation Operation
        x = torch.cat([Fi, F_spf_i_minus_1], dim=1)

        # Convolution Block x 2
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        _x = x.clone()

        # Resize Operation
        if self.resize_type == 'upsample':
            x = self.upsample(x)
        elif self.resize_type == 'downsample':
            x = self.downsample_conv(x)
        elif self.resize_type == 'none':
            x = self.identity_conv(x)

        # Semantic Selection Attention
        attn = self.global_avg_pool(_x)
        attn = attn.view(attn.size(0), -1)
        attn = self.gelu(self.fc1(attn))
        attn = self.sigmoid(self.fc2(attn))
        attn = attn.view(attn.size(0), attn.size(1), 1, 1)

        F_spf_i = x * attn

        return F_spf_i


# Example of usage
if __name__ == "__main__":
    # Dummy input tensors
    Fi = torch.randn(1, 64, 128, 128)  # Current feature map
    F_spf_i_minus_1 = torch.randn(1, 64, 128, 128)  # Previous SPF output

    # Create SPF unit with different resize types
    spf_unit_none = SPFUnit(in_channels=128, out_channels=64, resize_type='none')
    spf_unit_upsample = SPFUnit(in_channels=128, out_channels=64, resize_type='upsample')
    spf_unit_downsample = SPFUnit(in_channels=128, out_channels=64, resize_type='downsample')

    # Forward pass
    output_none = spf_unit_none(Fi, F_spf_i_minus_1)
    output_upsample = spf_unit_upsample(Fi, F_spf_i_minus_1)
    output_downsample = spf_unit_downsample(Fi, F_spf_i_minus_1)

    print("Output shape with no resize:", output_none.shape)
    print("Output shape with upsample:", output_upsample.shape)
    print("Output shape with downsample:", output_downsample.shape)
