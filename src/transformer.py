"""
Transformer Network (Generator) for Fast Neural Style Transfer
Feed-forward CNN with residual blocks for real-time style transfer
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization and ReLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential()
        
        if upsample:
            self.block.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        
        self.block.add_module('reflection_pad', nn.ReflectionPad2d(kernel_size // 2))
        self.block.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        
        if normalize:
            self.block.add_module('instance_norm', nn.InstanceNorm2d(out_channels))
        
        if relu:
            self.block.add_module('relu', nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False)
        )
    
    def forward(self, x):
        return x + self.block(x)


class TransformerNet(nn.Module):
    """
    Fast Neural Style Transfer Transformer Network
    Architecture: Encoder -> Residual Blocks -> Decoder
    """
    
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
        )
        
        # Residual blocks (transformation)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, normalize=False, relu=False),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


def test_transformer():
    """Test the transformer network with a dummy input"""
    model = TransformerNet()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_transformer()
