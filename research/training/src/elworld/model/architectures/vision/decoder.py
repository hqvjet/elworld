import torch
import torch.nn as nn

from elworld.model.architectures.vision.residual_stack import ResidualStack

class VisionDecoder(nn.Module):
    def __init__(
        self, input_channels=3, num_hidden=128, res_layer=2, res_hidden=32, output_channels=3
    ):
        super(VisionDecoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_hidden, kernel_size=3, stride=1, padding=1) # [B, 64, 24, 32] -> [B, 128, 24, 32]
        self.res_stack = ResidualStack(
            input_channels=num_hidden, num_hidden=num_hidden, 
            res_layer=res_layer, res_hidden=res_hidden
        ) # [B, 128, 24, 32] -> [B, 128, 24, 32]
        self.conv2 = nn.ConvTranspose2d(num_hidden, num_hidden//2, kernel_size=4, stride=2, padding=1) # [B, 128, 24, 32] -> [B, 64, 48, 64]
        self.conv3 = nn.ConvTranspose2d(num_hidden//2, input_channels, kernel_size=4, stride=2, padding=1) # [B, 64, 48, 64] -> [B, 3, 96, 128]
        self.conv4 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1) # [B, 3, 96, 128] -> [B, 3, 192, 256]

    def forward(self, x):
        x= self.conv1(x)
        x = torch.relu(x)
        x = self.res_stack(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        return x