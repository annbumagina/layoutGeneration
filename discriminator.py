import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 256->128
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1),  # 128->64
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64->32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32->16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16->8
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 8->4
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)  # 4x4x1
        )

    def forward(self, images):
        return self.main(images).squeeze()
