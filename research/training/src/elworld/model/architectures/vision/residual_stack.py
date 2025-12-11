import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels=128, num_hidden=128, res_hidden=32):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, res_hidden, kernel_size=3, stride=1, padding=1, bias=False), # [B, 128, 24, 32] -> [B, 32, 24, 32]
            nn.ReLU(inplace=False),
            nn.Conv2d(res_hidden, num_hidden, kernel_size=1, stride=1, bias=False) # [B, 32, 24, 32] -> [B, 128, 24, 32]
        )

    def forward(self, x):
        return x + self.block(x) # Sum of each element, not matrix multiplication


class ResidualStack(nn.Module):
    def __init__(
        self, input_channels=3, num_hidden=128, res_layer=2, res_hidden=32,
    ):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(input_channels, num_hidden, res_hidden) # [B, 128, 24, 32] -> [B, 128, 24, 32]: each element may change due to vector addition
                for _ in range(res_layer) # repeat residual block
            ]
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) #  x = x + block1 + block2
        return self.relu(x)