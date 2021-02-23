from math import ceil, floor
from pycocotools import coco
import numpy as np
import torch
from matplotlib import pyplot as plt
from functools import partial
from PIL import Image


def is_in_image(img, ann):
    y_i, x_i, w, h = ann['bbox']
    delta_w = (img['width'] - 256) / 2.
    delta_h = (img['height'] - 256) / 2.
    x_c = x_i + h / 2.
    y_c = y_i + w / 2.
    return delta_h < x_c < 256 + delta_h and delta_w < y_c < 256 + delta_w


class COCODataset():
    def __init__(self, thing_file, stuff_file=None, filter_cats=None, transform=None):
        """
        Args:
            thing_file (string): File with all the thing images' regions.
            stuff_file (string): File with all the stuff images' regions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cat_num = 92
        self.thing_coco_api = coco.COCO(annotation_file=thing_file)
        self.stuff_coco_api = None
        if stuff_file is not None:
            self.stuff_coco_api = coco.COCO(annotation_file=stuff_file)

        self.ann_ids_cat = []
        abc = []
        for cat in range(self.cat_num):
            ann_ids = self.thing_coco_api.getAnnIds(catIds=[cat])
            abc.append(len(ann_ids))
            self.ann_ids_cat.append(ann_ids)
        self.filter_cats = np.nonzero(abc)[0]
        self.img_ids = self.thing_coco_api.getImgIds()
        #self.img_ids = list(filter(lambda img_id: len(self.thing_coco_api.getAnnIds(img_id)) >= 2, self.img_ids))

        # both sides greater than 256
        self.img_ids = list(filter(lambda img_id: self.thing_coco_api.loadImgs([img_id])[0]['width'] >= 256 and
                                                  self.thing_coco_api.loadImgs([img_id])[0]['height'] >= 256,
                                   self.img_ids))
        # contain filter categories
        if filter_cats is not None:
            self.filter_cats = filter_cats
            self.img_ids = list(filter(partial(self.contain_cats, filter_cats), self.img_ids))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids = np.take(self.img_ids, idx)
        return ids

    def contain_cats(self, cats, img_id):
        anns = self.thing_coco_api.loadAnns(self.thing_coco_api.getAnnIds(imgIds=[img_id]))
        for ann in anns:
            if ann['category_id'] in cats:
                return True
        return False

    def rand_obj_by_img(self, ids):
        """
        Chooses one random object on each image.

        Args:
            ids (list or tensor): Image ids.
        Returns:
            x (list): x coordinates of objects (vertical)
            y (list): y coordinates of objects (horizontal)
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
            annotation_ids = self.thing_coco_api.getAnnIds(imgIds=img_id, catIds=self.filter_cats)
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
            annotations = self.thing_coco_api.loadAnns(annotation_ids)
            if self.stuff_coco_api is not None:
                stuff_ids = self.stuff_coco_api.getAnnIds(img_id)
                annotations = annotations + self.stuff_coco_api.loadAnns(stuff_ids)

            cat_id = annotations[0]["category_id"]
            mask = self.thing_coco_api.annToMask(annotations[0])
            mask[mask == 1] = cat_id
            segm = mask
            for i in range(1, len(annotations)):
                cat_id = annotations[i]["category_id"]
                mask = self.thing_coco_api.annToMask(annotations[i])
                segm[np.logical_and(mask == 1, segm == 0)] = cat_id

            segm = self.crop(segm, 256, 256)
            images.append(segm)
        return images

    def get_cat_num(self):
        return self.cat_num

    def crop_by_bbox(mask, bbox):
        x1, y1, x2, y2 = bbox
        x1 = floor(x1)
        y1 = floor(y1)
        x2 = ceil(x2)
        y2 = ceil(y2)
        return mask[y1:y2 + 1, x1:x2 + 1]

    def get_trimap_by_id(self, id):
        ann = self.thing_coco_api.loadAnns([id])[0]
        return self.thing_coco_api.annToMask(ann)

    def get_bbox_by_id(self, id):
        ann = self.thing_coco_api.loadAnns([id])[0]
        x1, y1, w, h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def get_objects_by_ids(self, ids):
        objects = []
        anns = self.thing_coco_api.loadAnns(ids)
        for i in range(len(ids)):
            data = self.thing_coco_api.annToMask(anns[i])
            data = data[~np.all(data == 0, axis=1)]
            idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
            data = np.delete(data, idx, axis=1)
            cat_id = anns[i]["category_id"]
            if data.size == 0:
                data = np.array([[cat_id]])
            data[data > 0] = cat_id
            objects.append(data)
        return objects

    def get_rand_by_classes(self, cats):
        """
        Chooses random objects from given categories.

        Args:
            cats (list or tensor): Categories indexes.
        Returns:
            objects (numpy array): Ids of chosen objects
        """
        if torch.is_tensor(cats):
            cats = cats.numpy()
        distinct = list(set(cats))

        objects = np.empty(len(cats))
        for cat in distinct:
            cat_ind = np.where(cats == cat)[0]
            cnt = len(cat_ind)
            ann_ids = self.ann_ids_cat[cat]
            ann_ids_ids = np.random.randint(0, len(ann_ids), cnt)
            objects[cats == cat] = np.asarray(ann_ids)[ann_ids_ids]
        return objects

    def get_rand_classes(self, cnt):
        cats = np.random.randint(0, len(self.filter_cats), cnt)
        return np.take(self.filter_cats, cats)

    def get_ratios(self, objects):
        """
        Computes ratios (height / width) of given objects
        """
        ratios = np.empty(len(objects))
        for i in range(len(objects)):
            a, b = objects[i].shape
            ratios[i] = a / b
        return ratios

    def get_img(self, img_id):
        img_file = self.thing_coco_api.loadImgs([img_id])[0]['file_name']
        self.thing_coco_api.download("coco/images", [img_id])
        img = Image.open("coco/images/" + img_file).convert("RGB")
        return np.array(img) / 255.0

    def get_img_by_obj(self, obj):
        ann = self.thing_coco_api.loadAnns([obj])[0]
        img_id = ann["image_id"]
        return self.get_img(img_id)

    def prepare_to_inpaint(self, object_i, x_i, y_i, h_i, w_i, segm_i):
        # resize object
        resized = torch.as_tensor(object_i).unsqueeze(0).unsqueeze(0)
        H = object_i.shape[0]
        W = object_i.shape[1]
        h = int(round(h_i))  # sizes
        w = int(round(w_i))  # coef
        iw = torch.linspace(0, W - 1, w).long()
        ih = torch.linspace(0, H - 1, h).long()
        resized = resized[:, :, ih[:, None], iw].squeeze().numpy()
        if resized.ndim < 2:
            print("too small")
            return np.array([]), 0, 0, 0, 0

        # crop object if needed
        h_i = resized.shape[0]
        w_i = resized.shape[1]
        x_i_1 = int(round(x_i)) - h_i // 2
        if x_i_1 < 0:
            resized = resized[-x_i_1:]
            x_i_1 = 0
        x_i_2 = int(round(x_i)) + (h_i + 1) // 2
        if x_i_2 > segm_i.shape[0]:
            resized = resized[:segm_i.shape[0] - x_i_2]
            x_i_2 = segm_i.shape[0]
        y_i_1 = int(round(y_i)) - w_i // 2
        if y_i_1 < 0:
            resized = resized[:, -y_i_1:]
            y_i_1 = 0
        y_i_2 = int(round(y_i)) + (w_i + 1) // 2
        if y_i_2 > segm_i.shape[1]:
            resized = resized[:, :segm_i.shape[1] - y_i_2]
            y_i_2 = segm_i.shape[1]
        return resized, x_i_1, x_i_2, y_i_1, y_i_2

    def inpaint_segmentation(self, object_i, x_i, y_i, h_i, w_i, segm_i):
        resized, x_i_1, x_i_2, y_i_1, y_i_2 = self.prepare_to_inpaint(object_i, x_i, y_i, h_i, w_i, segm_i)
        if resized.size == 0:
            print("bad size")
        elif not np.any(resized > 0):
            print("all false")
        (segm_i[x_i_1:x_i_2, y_i_1:y_i_2])[resized > 0] = resized[resized > 0]

    def inpaint_image(self, object_i, x_i, y_i, h_i, w_i, segm_i):
        resized, x_i_1, x_i_2, y_i_1, y_i_2 = self.prepare_to_inpaint(object_i, x_i, y_i, h_i, w_i, segm_i)
        if resized.size == 0:
            print("bad size")
        elif not np.any(resized > 0):
            print("all false")
        (segm_i[x_i_1:x_i_2, y_i_1:y_i_2])[resized[:, :, 3] > 0] = (resized[resized[:, :, 3] > 0])[:, :3]


# dataset = COCODataset('coco/annotations/instances_train2017.json', filter_cats=[1])
# print(len(dataset))
# backgrounds = dataset.__getitem__([0, 1, 2, 3, 4])
# ids = dataset.get_rand_by_cats([0, 0, 0, 0, 0])
# for i in range(len(ids)):
#     img = dataset.get_img_by_obj(ids[i])
#     co = cutout(img, dataset.get_trimap_by_id(ids[i]), dataset.get_bbox_by_id(ids[i]))
#     background = dataset.get_img(backgrounds[i])
#     plt.imshow(background)
#     plt.show()
#     dataset.inpaint_image(co, 50, 50, co.shape[0], co.shape[1], background)
#     plt.imshow(background)
#     plt.show()
