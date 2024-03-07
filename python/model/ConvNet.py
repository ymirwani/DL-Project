import torch.nn as nn
import torch

class ConvNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()