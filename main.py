import copy
import os

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from generator import Generator
from objects_dataset import BackgroundDataset, PeopleDataset

###########################
b_size = 16
load_file = 'models/checkpoint4.pth.tar'
###########################

dataset = BackgroundDataset(filter_cats=[1], image_dir='coco/images/train2017', transform=T.Compose([T.Resize(256), T.RandomCrop(256), T.ToTensor()]))
objects_dataset = PeopleDataset(image_dir='coco/images/fashionPNG', transform=T.Compose([T.Resize(256), T.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=0, drop_last=True)

device = torch.device("cuda")
netG = Generator().to(device)

if load_file is not None:
    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file)
        netG.load_state_dict(checkpoint['netG'])
        netG.eval()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(load_file))


with torch.no_grad():
    for i_batch, segms in enumerate(dataloader, 0):

        # Get Data
        segms = segms.to(device)
        objects = objects_dataset.get_rand_objects(b_size).to(device)

        # Generate
        noise = torch.randn(b_size, 100).to(device)
        theta_out = netG(segms, objects, noise)
        theta = theta_out + torch.FloatTensor([[1, 0, 0], [0, 1, 0]]).to(device)

        # Merge masks
        grid = F.affine_grid(theta, objects.size())
        obj_fake = F.grid_sample(objects, grid)
        color, mask = obj_fake[:, :3, :, :], obj_fake[:, 3:, :, :]
        fake = color * mask + segms * (1 - mask)

        for i in range(b_size):
            plt.imshow(segms.detach().cpu()[i].permute(1, 2, 0).numpy())
            plt.show()
            plt.imshow(objects.detach().cpu()[i].permute(1, 2, 0).numpy())
            plt.show()
            plt.imshow(fake.detach().cpu()[i].permute(1, 2, 0).numpy())
            plt.show()
