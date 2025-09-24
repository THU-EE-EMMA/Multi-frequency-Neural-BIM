import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for Conv2d, Linear, and BatchNorm2d layers.

    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    """
    Creates a 3x3 convolutional layer with padding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride. Defaults to 1.

    Returns:
        A Conv2d layer with 3x3 kernel, specified stride, and padding=1.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)