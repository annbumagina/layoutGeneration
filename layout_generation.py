import os
import shutil

from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from generator import *
from discriminator import *
from objects_dataset import BackgroundDataset, ObjectsDataset, PeopleDataset
import torch.nn.functional as F
import copy
from matplotlib import pyplot as plt
import numpy as np

torch.set_printoptions(threshold=10000000)
###########################
start_epoch = 0
num_epochs = 20
lr = 0.0002
beta1 = 0.5
b_size = 16
grad_lambda = 10.0
upd_lambda = 0.8
upd_lambda_big = 100.0
train_D = 2
load_file = 'models/checkpoint31.pth.tar'
min_object = 500.
###########################


def save_checkpoint(state, filename):
    torch.save(state, filename)


dataset = BackgroundDataset(filter_cats=[1], image_dir='coco/images/train2017', transform=T.Compose([T.Resize(256), T.RandomCrop(256), T.ToTensor()]))
objects_dataset = PeopleDataset(image_dir='coco/images/objects', transform=T.Compose([T.Resize(256), T.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=b_size*3, shuffle=True, num_workers=0, drop_last=True)

device = torch.device("cuda")
netG = Generator().to(device)
netD = Discriminator().to(device)
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

if load_file is not None:
    if os.path.isfile(load_file):
        print("=> loading checkpoint '{}'".format(load_file))
        checkpoint = torch.load(load_file)
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        netG.train()
        netD.train()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(load_file))


def get_upd_lambda(objects):  # big lambda for empty objects
    llambda = torch.full((b_size,), upd_lambda).to(device)
    for i in range(b_size):
        object = objects[i]
        mask = object[3:, :, :]
        if torch.sum(mask) < min_object:
            llambda[i] = upd_lambda_big
    return llambda


def gradient_penalty(prob, input_image):
    gradients = torch.autograd.grad(outputs=prob, inputs=input_image,
                                    grad_outputs=torch.ones(prob.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(b_size, -1)
    slopes = torch.sqrt(torch.sum(gradients ** 2 + 1e-8, dim=1))
    return ((slopes - 1.) ** 2).mean()


critic_real = []
critic_fake = []
gen_loss = []

print(len(dataset))
print("Starting Training Loop...")
for epoch in range(start_epoch, num_epochs + start_epoch):
    for i_batch, segms in enumerate(dataloader, 0):

        # Get Data
        segms1 = segms[:b_size].to(device)
        segms2 = segms[b_size: b_size*2].to(device)
        segms3 = segms[b_size*2:].to(device)
        objects1 = objects_dataset.get_rand_objects(b_size).to(device)
        objects2 = objects_dataset.get_rand_objects(b_size).to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        for i in range(train_D):
            ## Train with all-real batch
            output_real = netD(segms1)

            ## Train with all-fake batch
            noise = torch.randn(b_size, 100).to(device)
            theta_out = netG(segms2, objects1, noise)
            theta = theta_out + torch.FloatTensor([[1, 0, 0], [0, 1, 0]]).to(device)
            theta = theta ** torch.FloatTensor([[-1, 1, 1], [1, -1, 1]]).to(device)
            # Merge masks
            grid = F.affine_grid(theta, objects1.size())
            obj_fake = F.grid_sample(objects1, grid)
            color, mask = obj_fake[:, :3, :, :], obj_fake[:, 3:, :, :]
            fake = color * mask + segms2 * (1 - mask)
            # Classify all fake batch with D
            output_fake = netD(fake)

            ## Train interpolated
            alpha = torch.rand(b_size, 1, 1, 1).cuda()
            interpolated = alpha * segms1 + (1 - alpha) * fake
            interpolated = torch.autograd.Variable(interpolated, requires_grad=True).cuda()
            prob_interpolated = netD(interpolated)

            # Calculate D's loss -1=real 1=fake
            loss_grad = gradient_penalty(prob_interpolated, interpolated)
            loss_disc = -torch.mean(output_real) + torch.mean(output_fake) + grad_lambda * loss_grad
            #print(-torch.mean(output_real) + torch.mean(output_fake), grad_lambda * loss_grad)

            if i == train_D - 1:
                critic_real.append(-torch.mean(output_real).item())
                critic_fake.append(torch.mean(output_fake).item())
            netD.zero_grad()
            loss_disc.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        noise = torch.randn(b_size, 100).to(device)
        theta_out = netG(segms3, objects2, noise)
        theta = theta_out + torch.FloatTensor([[1, 0, 0], [0, 1, 0]]).to(device)
        theta = theta ** torch.FloatTensor([[-1, 1, 1], [1, -1, 1]]).to(device)
        # Merge masks
        grid = F.affine_grid(theta, objects2.size())
        obj_fake = F.grid_sample(objects2, grid)
        color, mask = obj_fake[:, :3, :, :], obj_fake[:, 3:, :, :]
        fake = color * mask + segms3 * (1 - mask)
        # Classify all fake batch with D
        output = netD(fake)

        # Calculate G's loss
        loss_upd = torch.mean(get_upd_lambda(obj_fake) * torch.sum(theta_out.view(b_size, -1) ** 2 + 1e-8, dim=-1))
        loss_gen = -torch.mean(output) + loss_upd

        gen_loss.append(loss_gen.item())
        netG.zero_grad()
        loss_gen.backward()
        optimizerG.step()

        # Output training stats
        if (i_batch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs + start_epoch}] Batch {i_batch + 1}/{len(dataloader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            if (i_batch + 1) % 440 == 0:
                x = list(range(0, len(critic_real)))
                plt.plot(x, critic_real, label='crit_real')
                plt.plot(x, critic_fake, label='crit_fake')
                plt.plot(x, gen_loss, label='gen')
                plt.legend()
                plt.savefig("images/" + str(epoch) + "_" + str(i_batch + 1) + "loss.png")
                plt.close()
                print(loss_upd)
                print(theta)
            if (i_batch + 1) % 440 == 0:
                for i in range(min(10, b_size)):
                    # plt.imshow(segms3.detach().cpu().squeeze().numpy()[i])
                    # plt.show()
                    # plt.imshow(fake_segm.detach().cpu().squeeze().numpy()[i])
                    # plt.show()
                    plt.imsave("images/" + str(epoch) + "_" + str(i_batch + 1) + "_" + str(i) + "real.png",
                               segms3.detach().cpu()[i].permute(1, 2, 0).numpy())
                    plt.imsave("images/" + str(epoch) + "_" + str(i_batch + 1) + "_" + str(i) + "fake.png",
                               fake.detach().cpu()[i].permute(1, 2, 0).numpy())

    save_checkpoint({
        'epoch': epoch + 1,
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
    }, 'models/checkpoint' + str(epoch) + '.pth.tar')
    print("saved")
