import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, classes):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(  # 256->128
                3, 64, kernel_size=4,
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
        self.lin = nn.Linear(100, 256 * 256)
        self.embed = nn.Sequential(
            nn.Embedding(classes, 50),
            nn.Linear(50, 256 * 256)
        )
        self.final = nn.Sequential(
            nn.Linear(1025, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()  # x, y, coef
        )

    def forward(self, noise, classes, ratio, images):
        noise = self.lin(noise)
        noise = noise.view(-1, 256, 256)
        classes = self.embed(classes)
        classes = classes.view(-1, 256, 256)
        x = torch.cat([noise.unsqueeze(1), classes.unsqueeze(1), images.unsqueeze(1)], dim=1)
        vec = self.main(x).squeeze()
        vec = torch.cat([vec, ratio.unsqueeze(1)], dim=1)
        return self.final(vec)
