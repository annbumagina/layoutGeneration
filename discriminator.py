import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(  # 256->128
                1, 64, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 128->64
                64, 128, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 64->32
                128, 256, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 32->16
                256, 512, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 16->8
                512, 512, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 8->4
                512, 512, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(  # 1024x1x1
                512, 1024, kernel_size=4,
                stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.final = nn.Sequential(
            nn.Linear(1027, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, x, y, coef):
        main = self.main(images.unsqueeze(1))
        main = torch.cat([main.squeeze(), x.unsqueeze(1), y.unsqueeze(1), coef.unsqueeze(1)], dim=1)
        return self.final(main)
