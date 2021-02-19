import copy
from matplotlib import pyplot as plt
import torch

from cutter import cutout
from generator import Generator
from objects_dataset import COCODataset

device = torch.device("cuda")
dataset = COCODataset('coco/annotations/instances_train2017.json')
netG = Generator(dataset.get_cat_num()).to(device)
netG.load_state_dict(torch.load("models/4netG.pt"))

netG.eval()
with torch.no_grad():
    data = dataset.__getitem__([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    segms = dataset.segm_by_ids(data)
    noise = torch.randn(len(data), 100)
    classes = torch.full((len(data),), 1, dtype=torch.int)
    # Generate fake image batch with G
    ids = dataset.get_rand_by_classes(classes)
    objects = dataset.get_objects_by_ids(ids)
    ratios = dataset.get_ratios(objects)
    fake = netG(noise.to(device), classes.to(device), torch.as_tensor(ratios).type(torch.FloatTensor).to(device),
                torch.as_tensor(segms).type(torch.FloatTensor).to(device))

    # merge masks
    prev_segm = copy.deepcopy(segms)
    abc = fake.detach().cpu().numpy()
    x = abc[:, 0] * 255
    y = abc[:, 1] * 255
    coef = abc[:, 2] * 246 + 10
    print(x)
    print(y)
    print(coef)

    sizes = coef * ratios
    for i in range(len(data)):
        if round(sizes[i]) > 0 and round(coef[i]) > 0:
            img = dataset.get_img_by_obj(ids[i])
            co = cutout(img, dataset.get_trimap_by_id(ids[i]), dataset.get_bbox_by_id(ids[i]))
            background = dataset.get_img(data[i])
            dataset.inpaint_image(co, x[i], y[i], sizes[i], coef[i], background)
    for i in range(len(data)):
        plt.imshow(prev_segm[i])
        plt.show()
        plt.imshow(segms[i])
        plt.show()
        #plt.imsave("images/" + str(i) + "_test_prevSegm.png", prev_segm[i])
        #plt.imsave("images/" + str(i) + "_test_segm.png", segms[i])