from models.network.hypercomplex_layers import *

def count_phmlinear(m, x, y):
    # x[0]: 输入，y: 输出
    batch = x[0].shape[0]
    in_features = m.in_features
    out_features = m.out_features

    # 标准 Linear FLOPs
    flops_linear = batch * in_features * out_features

    # 估计 Kronecker Product 部分 FLOPs：
    # 这里的计算需要根据 A, S 的尺寸来手动推导
    n = m.n
    kron_flops = n * n * n * ((out_features // n) * (in_features // n))

    total_flops = flops_linear + kron_flops
    m.total_ops = torch.DoubleTensor([int(total_flops)])


def count_phconv(m, x, y):
    batch = x[0].shape[0]
    in_channels = m.in_features
    out_channels = m.out_features
    kernel_h = m.kernel_size
    kernel_w = m.kernel_size
    out_h = y.shape[2]
    out_w = y.shape[3]

    # 标准 Conv2d FLOPs
    flops_conv = batch * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w

    # 估计 Kronecker Product 部分 FLOPs：
    n = m.n
    # 例如：对 F 的计算 FLOPs
    kron_flops = n * n * n * ((out_channels // n) * (in_channels // n) * kernel_h * kernel_w)

    total_flops = flops_conv + kron_flops
    m.total_ops = torch.DoubleTensor([int(total_flops)])

