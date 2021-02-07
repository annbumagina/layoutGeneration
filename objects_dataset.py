from pycocotools import coco
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from functools import partial


def is_in_image(img, ann):
    y_i, x_i, w, h = ann['bbox']
    delta_w = (img['width'] - 256) / 2.
    delta_h = (img['height'] - 256) / 2.
    x_c = x_i + h / 2.
    y_c = y_i + w / 2.
    return delta_h < x_c < 256 + delta_h and delta_w < y_c < 256 + delta_w


class COCODataset():
    def __init__(self, thing_file, stuff_file, transform=None):
        """
        Args:
            thing_file (string): File with all the thing images' regions.
            stuff_file (string): File with all the stuff images' regions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cat_num = 92
        self.thing_coco_api = coco.COCO(annotation_file=thing_file)
        self.stuff_coco_api = coco.COCO(annotation_file=stuff_file)

        abc = []
        for cat in range(self.cat_num):
            ann_ids = self.thing_coco_api.getAnnIds(catIds=[cat])
            abc.append(len(ann_ids))
        self.classes = np.nonzero(abc)[0]
        self.img_ids = list(filter(lambda img_id: len(self.thing_coco_api.getAnnIds(img_id)) >= 2, self.thing_coco_api.getImgIds()))
        self.img_ids = list(filter(lambda img_id: self.thing_coco_api.loadImgs([img_id])[0]['width'] >= 256 and self.thing_coco_api.loadImgs([img_id])[0]['height'] >= 256, self.img_ids))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids = np.take(self.img_ids, idx)
        return ids

    def rand_obj_by_id(self, ids):
        """
        Chooses one random object on each image.

        Args:
            ids (list or tensor): Image ids.
        Returns:
            x (list): x coordinates of objects
            y (list): y coordinates of objects
            widths (list): coefficients of objects
        """
        if torch.is_tensor(ids):
            ids = ids.tolist()
        imgs = self.thing_coco_api.loadImgs(ids)
        x = []
        y = []
        widths = []
        for i in range(len(ids)):
            img_id = ids[i]
            img = imgs[i]
            annotation_ids = self.thing_coco_api.getAnnIds(img_id, catIds=self.classes)
            annotations = self.thing_coco_api.loadAnns(annotation_ids)
            annotations = list(filter(partial(is_in_image, img), annotations))
            if len(annotations) == 0:
                x.append(0.5)
                y.append(0.5)
                widths.append(0.5)
            else:
                y_i, x_i, w, h = np.random.choice(annotations)['bbox']
                delta_w = (img['width'] - 256) / 2.
                delta_h = (img['height'] - 256) / 2.
                x_c = x_i + h / 2. - delta_h
                y_c = y_i + w / 2. - delta_w
                x.append(x_c / 256.)
                y.append(y_c / 256.)
                widths.append(w / img['width'])
        return x, y, widths

    def crop(self, img, tw, th):
        """
        Crops array
        """
        w, h = img.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[x1:x1 + tw, y1:y1 + th]

    def segm_by_ids(self, ids):
        """
        Returns segmentations of given images.

        Args:
            ids (list or tensor): Image ids.
        Returns:
            images (list): segmentations
        """
        if torch.is_tensor(ids):
            ids = ids.tolist()
        images = []
        for img_id in ids:
            annotation_ids = self.thing_coco_api.getAnnIds(img_id)
            stuff_ids = self.stuff_coco_api.getAnnIds(img_id)
            annotations = self.thing_coco_api.loadAnns(annotation_ids) + self.stuff_coco_api.loadAnns(stuff_ids)

            cat_id = annotations[0]["category_id"]
            mask = self.thing_coco_api.annToMask(annotations[0])
            mask[mask == 1] = cat_id
            segm = mask
            for i in range(1, len(annotations)):
                cat_id = annotations[i]["category_id"]
                mask = self.thing_coco_api.annToMask(annotations[i])
                mask[mask == 1] = cat_id
                segm += mask

            segm = self.crop(segm, 256, 256)
            images.append(segm)
        return images

    def get_cat_num(self):
        return len(self.classes)

    def get_rand_by_cats(self, cats):
        """
        Chooses random objects from given categories.

        Args:
            cats (list or tensor): Categories indexes.
        Returns:
            objects (list of 2d numpy arrays): Chosen objects
        """
        if torch.is_tensor(cats):
            cats = cats.numpy()
        cats = np.take(self.classes, cats)
        #print(len(cats))

        objects = [None] * len(cats)
        for cat in self.classes:
            cat_ind = np.where(cats == cat)[0]
            cnt = len(cat_ind)
            if cnt > 0:
                ann_ids = self.thing_coco_api.getAnnIds(catIds=[cat])
                ann_ids_ids = np.random.randint(0, len(ann_ids), cnt)
                anns = self.thing_coco_api.loadAnns(np.asarray(ann_ids)[ann_ids_ids])

                for i in range(cnt):
                    data = self.thing_coco_api.annToMask(anns[i])
                    data = data[~np.all(data == 0, axis=1)]
                    idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
                    data = np.delete(data, idx, axis=1)
                    if data.size == 0:
                        data = np.array([[1]])
                    data[data > 0] = cat
                    objects[cat_ind[i]] = data
        return objects

    def get_ratios(self, objects):
        """
        Computes ratios (height / width) of given objects
        """
        ratios = np.empty(len(objects))
        for i in range(len(objects)):
            a, b = objects[i].shape
            ratios[i] = a / b
        return ratios


dataset = COCODataset('coco/annotations/instances_train2017.json', 'coco/annotations/stuff_train2017.json')
ids = dataset.__getitem__([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
segms = dataset.segm_by_ids(ids)
for i in range(10):
    plt.imshow(segms[i])
    plt.show()

# classes = torch.randint(1, dataset.get_cat_num(), (100,))
# objects = dataset.get_rand_by_cats(classes.numpy())
# ratios = dataset.get_ratios(objects)
