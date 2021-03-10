import torch
from torch import nn


class Inpainter(nn.Module):
    def __init__(self):
        super(Inpainter, self).__init__()
        self.upsample = nn.Upsample(scale_factor=256, mode='nearest')
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=4,
                      stride=2,
                      padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()
        )

    def forward(self, objects, x, y, coef):
        x = self.upsample(x.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        y = self.upsample(y.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        coef = self.upsample(coef.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        result = torch.cat([objects.unsqueeze(1), x, y, coef], dim=1)
        return self.main(result)
