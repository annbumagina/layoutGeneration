import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from inpainter import Inpainter
from objects_dataset import COCODataset

###########################
num_epochs = 10
lr = 1e-3
b_size = 16
###########################

dataset = COCODataset('coco/annotations/instances_train2017.json')#, 'coco/annotations/stuff_train2017.json')
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=0)

device = torch.device("cuda")
net = Inpainter().to(device)
# Setup Adam optimizers for both G and D
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

print(len(dataset))
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i_batch, data in enumerate(dataloader, 0):

        net.zero_grad()
        # Get Data
        #segms = dataset.segm_by_ids(data)
        classes = torch.as_tensor(dataset.get_rand_classes(len(data)))
        objects = dataset.get_objects_by_ids(dataset.get_rand_by_classes(classes))
        padded_objects = dataset.pad_objects(objects)
        ratios = dataset.get_ratios(objects)
        x = torch.rand(len(data))
        y = torch.rand(len(data))
        coef = torch.rand(len(data))
        # Net Inpaint
        output = net(torch.as_tensor(padded_objects).type(torch.FloatTensor).to(device),
                     x.to(device), y.to(device), coef.to(device)).squeeze(1)
        output = output
        # Self Inpaint
        x = x.numpy() * 255  # vertical
        y = y.numpy() * 255  # horizontal
        coef = coef.numpy() * 246 + 10  # width
        sizes = coef * ratios  # height
        emp = []
        for i in range(len(data)):
            emp.append(np.zeros((256, 256)))
            if round(sizes[i]) > 0 and round(coef[i]) > 0:
                dataset.inpaint_segmentation(objects[i], x[i], y[i], sizes[i], coef[i], emp[i])
        target = torch.as_tensor(emp).type(torch.FloatTensor).to(device) / dataset.get_cat_num()
        # Calculate loss
        loss = criterion(output.view(-1, 256 * 256), target.view(-1, 256 * 256))
        loss.backward()
        optimizer.step()

        # Output training stats
        if (i_batch + 1) % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, num_epochs, i_batch + 1, len(dataloader), loss.item()))
            if (i_batch + 1) % 2000 == 0:
                for i in range(min(5, len(data))):
                    # plt.imshow(target.cpu().numpy()[i])
                    # plt.show()
                    # plt.imshow(output.detach().cpu().numpy()[i])
                    # plt.show()
                    plt.imsave("images/" + str(epoch) + "_" + str(i_batch + 1) + "_" + str(i) + "target.png",
                               target.cpu().numpy()[i])
                    plt.imsave("images/" + str(epoch) + "_" + str(i_batch + 1) + "_" + str(i) + "out.png",
                               output.detach().cpu().numpy()[i])

    torch.save(net.state_dict(), "models/" + str(epoch) + "netInpainter.pt")
    print("saved")
