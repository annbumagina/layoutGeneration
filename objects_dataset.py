from pycocotools import coco
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class COCODataset():
    def __init__(self, annotation_file, transform=None):
        """
        Args:
            annotation_file (string): File with all the images' regions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cat_num = 92
        self.coco_api = coco.COCO(annotation_file=annotation_file)

        abc = []
        for cat in range(self.cat_num):
            ann_ids = self.coco_api.getAnnIds(catIds=[cat])
            abc.append(len(ann_ids))
        self.classes = np.nonzero(abc)[0]
        self.img_ids = list(filter(lambda img_id: len(self.coco_api.getAnnIds(img_id)) >= 2, self.coco_api.getImgIds()))
        self.img_ids = list(filter(lambda img_id: self.coco_api.loadImgs([img_id])[0]['width'] >= 256 and self.coco_api.loadImgs([img_id])[0]['height'] >= 256, self.img_ids))

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
        x = []
        y = []
        widths = []
        for img_id in ids:
            annotation_ids = self.coco_api.getAnnIds(img_id)
            annotations = self.coco_api.loadAnns(annotation_ids)
            x_i, y_i, w, h = np.random.choice(annotations)['bbox']
            x.append((x_i + w) / 2)
            y.append((y_i + h) / 2)
            widths.append(w)
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
            annotation_ids = self.coco_api.getAnnIds(img_id)
            annotations = self.coco_api.loadAnns(annotation_ids)

            cat_id = annotations[0]["category_id"]
            mask = self.coco_api.annToMask(annotations[0])
            mask[mask == 1] = cat_id
            segm = mask
            for i in range(1, len(annotations)):
                cat_id = annotations[i]["category_id"]
                mask = self.coco_api.annToMask(annotations[i])
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
        for cat in range(self.cat_num):
            cat_ind = np.where(cats == cat)[0]
            cnt = len(cat_ind)
            if cnt > 0:
                ann_ids = self.coco_api.getAnnIds(catIds=[cat])
                ann_ids_ids = np.random.randint(0, len(ann_ids), cnt)
                anns = self.coco_api.loadAnns(np.asarray(ann_ids)[ann_ids_ids])

                for i in range(cnt):
                    data = self.coco_api.annToMask(anns[i])
                    data = data[~np.all(data == 0, axis=1)]
                    idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
                    data = np.delete(data, idx, axis=1)
                    if data.size == 0:
                        data = np.array([[1]])
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


# dataset = COCODataset('coco/annotations/instances_train2017.json')
# dataset.segm_by_ids(dataset.__getitem__(list(range(100))))

# classes = torch.randint(1, dataset.get_cat_num(), (100,))
# objects = dataset.get_rand_by_cats(classes.numpy())
# ratios = dataset.get_ratios(objects)
