from fvcore.nn.jit_handles import get_shape
import numpy as np

def add_flop_jit(inputs, outputs):
    num_elements = np.prod(get_shape(outputs[0]))
    return num_elements

def mul_flop_jit(inputs, outputs):
    num_elements = np.prod(get_shape(outputs[0]))
    return num_elements

def div_flop_jit(inputs, outputs):
    num_elements = np.prod(get_shape(outputs[0]))
    return num_elements

def gelu_flop_jit(inputs, outputs):
    # GELU计算较为复杂，但简化为每个元素几个操作
    num_elements = np.prod(get_shape(outputs[0]))
    # GELU的计算涉及指数函数、除法等，这里假设大约为3次操作
    return num_elements * 3

def softmax_flop_jit(inputs, outputs):
    # Softmax包括指数运算、求和和除法
    num_elements = np.prod(get_shape(outputs[0]))
    # 简化计算，假设为每个元素3次操作
    return num_elements * 3

def tanh_flop_jit(inputs, outputs):
    num_elements = np.prod(get_shape(outputs[0]))
    # 双曲正切通常计2次操作（指数运算和除法）
    return num_elements * 2