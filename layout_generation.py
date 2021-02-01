from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from background_dataset import *
from generator import *
from discriminator import *
from objects_dataset import COCODataset
import torch.nn.functional as F
import copy
from matplotlib import pyplot as plt

###########################
num_epochs = 5
lr = 0.0002
beta1 = 0.5
b_size = 64
###########################

dataset = COCODataset('coco/annotations/instances_train2017.json')
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=0)

netG = Generator(dataset.get_cat_num())
netD = Discriminator()
criterion = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print(len(dataset))
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i_batch, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        label = torch.full((b_size,), real_label, dtype=torch.float)
        # Forward pass real batch through D
        segms = dataset.segm_by_ids(data)
        x_gen, y_gen, coefs = dataset.rand_obj_by_id(data)
        output = netD(torch.as_tensor(segms).type(torch.FloatTensor),
                      torch.as_tensor(x_gen),
                      torch.as_tensor(y_gen),
                      torch.as_tensor(coefs)).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100)
        classes = torch.randint(0, dataset.get_cat_num(), (b_size,))
        # Generate fake image batch with G
        objects = dataset.get_rand_by_cats(classes)
        ratios = dataset.get_ratios(objects)
        fake = netG(noise, classes, torch.as_tensor(ratios).type(torch.FloatTensor),
                    torch.as_tensor(segms).type(torch.FloatTensor))
        label.fill_(fake_label)

        ############# MERGE MASKS ################
        prev_segm = copy.deepcopy(segms)
        abc = fake.detach().numpy()
        print("merge")
        x = abc[:, 0] * 255
        y = abc[:, 1] * 255
        coef = abc[:, 2] * 256
        sizes = coef * ratios
        for i in range(b_size):
            resized = F.interpolate(torch.as_tensor(objects[i]).unsqueeze(0).unsqueeze(0),
                                    size=(round(sizes[i]), round(coef[i]))).squeeze().numpy()

            ###  crop object if needed
            h_i, w_i = resized.shape
            x_i_1 = round(x[i]) - h_i // 2
            if x_i_1 < 0:
                resized = resized[-x_i_1:]
                x_i_1 = 0
            x_i_2 = round(x[i]) + (h_i + 1) // 2
            if x_i_2 > segms[i].shape[0]:
                resized = resized[:segms[i].shape[0]-x_i_2]
                x_i_2 = segms[i].shape[0]
            y_i_1 = round(y[i]) - w_i // 2
            if y_i_1 < 0:
                resized = resized[:, -y_i_1:]
                y_i_1 = 0
            y_i_2 = round(y[i]) + (w_i + 1) // 2
            if y_i_2 > segms[i].shape[1]:
                resized = resized[:, :segms[i].shape[1]-y_i_2]
                y_i_2 = segms[i].shape[1]
            ###

            if resized.size == 0:
                print("bad size")
            if not np.any(resized > 0):
                print("all false")
            (segms[i][x_i_1:x_i_2, y_i_1:y_i_2])[resized > 0] = resized[resized > 0]
        ##########################################

        # Classify all fake batch with D
        output = netD(torch.as_tensor(segms).type(torch.FloatTensor),
                      fake.detach()[:, 0], fake.detach()[:, 1], fake.detach()[:, 2]).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(torch.as_tensor(segms).type(torch.FloatTensor), fake[:, 0], fake[:, 1], fake[:, 2]).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i_batch % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i_batch, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            for i in range(5):
                plt.imshow(prev_segm[i])
                plt.show()
                plt.imshow(segms[i])
                plt.show()
