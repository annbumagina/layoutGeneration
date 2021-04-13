import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)

    def forward(self, features, images):
        features = self.main(features)
        images = self.downsample(images)
        features = torch.cat([features, images], dim=1)
        return features, images


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = Block(8, 16)  # 256->128
        self.block2 = Block(23, 32)  # 128->64
        self.block3 = Block(39, 64)  # 64->32
        self.block4 = Block(71, 128)  # 32->16
        self.block5 = Block(135, 256)  # 16->8
        self.block6 = Block(263, 512)  # 8->4

        self.lin = nn.Linear(100, 256 * 256)
        self.final = nn.Sequential(
            nn.Linear(519*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, images, objects, noise):
        noise = self.lin(noise)
        noise = noise.view(-1, 256, 256).unsqueeze(1)
        x = torch.cat([images, objects, noise], dim=1)

        images = torch.cat([images, objects], dim=1)

        features, images = self.block1(x, images)
        features, images = self.block2(features, images)
        features, images = self.block3(features, images)
        features, images = self.block4(features, images)
        features, images = self.block5(features, images)
        features, images = self.block6(features, images)

        features = features.view(-1, 519*4*4)
        return self.final(features).view(-1, 2, 3)
