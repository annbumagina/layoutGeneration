import copy
from math import ceil, floor
from pycocotools import coco
import numpy as np
import torch
from matplotlib import pyplot as plt
from functools import partial
from PIL import Image
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F
import cv2 as cv

from torchvision import transforms as T

from cutter import cutout

torch.set_printoptions(threshold=10000000)
thing_coco_api = coco.COCO(annotation_file='coco/annotations/instances_train2017.json')
stuff_coco_api = coco.COCO(annotation_file='coco/annotations/stuff_train2017.json')

# outdoor
sky = [157, 106]
ground = [126, 145, 144, 147, 140, 149, 125, 136, 111, 159, 154]
# indoor
floors = [114, 115, 116, 117, 118, 101]
wall = [171, 172, 173, 174, 175, 176, 177]


def is_in_image(img, ann):
    y_i, x_i, w, h = ann['bbox']
    delta_w = (img['width'] - 256) / 2.
    delta_h = (img['height'] - 256) / 2.
    x_c = x_i + h / 2.
    y_c = y_i + w / 2.
    return delta_h < x_c < 256 + delta_h and delta_w < y_c < 256 + delta_w


class BackgroundDataset:
    def __init__(self, image_dir, transform=None, filter_cats=None):
        """
        Args:
            filter_cats (list): categories to use.
        """
        self.cat_num = 92
        self.transform = transform
        self.image_dir = image_dir

        # self.ann_ids_cat = []
        # abc = []
        # for cat in range(self.cat_num):
        #     ann_ids = thing_coco_api.getAnnIds(catIds=[cat])
        #     abc.append(len(ann_ids))
        #     self.ann_ids_cat.append(ann_ids)
        # self.filter_cats = np.nonzero(abc)[0]
        self.img_ids = stuff_coco_api.getImgIds()
        self.img_ids = \
            list(filter(lambda img_id: (self.contain_cats(sky, img_id) and self.contain_cats(ground, img_id)) or
                                       (self.contain_cats(wall, img_id) and self.contain_cats(floors, img_id)),
                        self.img_ids))
        print(len(self.img_ids))
        # self.img_ids = list(filter(lambda img_id: len(thing_coco_api.getAnnIds(img_id)) >= 2, self.img_ids))
        # both sides greater than 256
        # self.img_ids = list(filter(lambda img_id: thing_coco_api.loadImgs([img_id])[0]['width'] >= 256 and
        #                                           thing_coco_api.loadImgs([img_id])[0]['height'] >= 256,
        #                            self.img_ids))
        # contain filter categories
        # if filter_cats is not None:
        #     self.filter_cats = filter_cats
        #     self.img_ids = list(filter(partial(self.contain_cats, filter_cats), self.img_ids))

    def __len__(self):
        return len(self.img_ids)

    def get_by_idx(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids = np.take(self.img_ids, idx)
        images = []
        for img_id in ids:
            img = self.get_img(img_id)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)
        return images

    def __getitem__(self, idx):
        img = self.get_img(self.img_ids[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def contain_cats(self, cats, img_id, stuff=True):
        if stuff:
            anns = stuff_coco_api.loadAnns(stuff_coco_api.getAnnIds(imgIds=[img_id]))
        else:
            anns = thing_coco_api.loadAnns(thing_coco_api.getAnnIds(imgIds=[img_id]))

        for ann in anns:
            if ann['category_id'] in cats:
                return True
        return False

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
            annotation_ids = thing_coco_api.getAnnIds(img_id)
            annotations = thing_coco_api.loadAnns(annotation_ids)
            if stuff_coco_api is not None:
                stuff_ids = stuff_coco_api.getAnnIds(img_id)
                annotations = annotations + stuff_coco_api.loadAnns(stuff_ids)

            cat_id = annotations[0]["category_id"]
            mask = thing_coco_api.annToMask(annotations[0])
            mask[mask == 1] = cat_id
            segm = mask
            for i in range(1, len(annotations)):
                cat_id = annotations[i]["category_id"]
                mask = thing_coco_api.annToMask(annotations[i])
                segm[np.logical_and(mask == 1, segm == 0)] = cat_id

            segm = self.crop(segm, 256, 256)
            images.append(segm)
        return images

    def get_cat_num(self):
        return self.cat_num

    def get_trimap_by_id(self, id):
        ann = thing_coco_api.loadAnns([id])[0]
        return thing_coco_api.annToMask(ann)

    def get_bbox_by_id(self, id):
        ann = thing_coco_api.loadAnns([id])[0]
        x1, y1, w, h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def get_img(self, img_id):
        img_file = thing_coco_api.loadImgs([img_id])[0]['file_name']
        # thing_coco_api.download("coco/images", [img_id])
        img = Image.open(self.image_dir + "/" + img_file).convert("RGB")
        return img #np.array(img) / 255.0

    def get_img_by_obj(self, obj):
        ann = thing_coco_api.loadAnns([obj])[0]
        img_id = ann["image_id"]
        return self.get_img(img_id)


class ObjectsDataset:
    def __init__(self, image_dir, filter_cats=None):
        """
        Args:
            filter_cats (list): Categories to use.
        """
        self.cat_num = 92
        self.ann_ids_cat = []
        self.image_dir = image_dir

        abc = []
        for cat in range(self.cat_num):
            ann_ids = thing_coco_api.getAnnIds(catIds=[cat])
            abc.append(len(ann_ids))
            self.ann_ids_cat.append(ann_ids)
        self.filter_cats = np.nonzero(abc)[0]

        # contain filter categories
        if filter_cats is not None:
            self.filter_cats = filter_cats
        self.annotations_ids = thing_coco_api.getAnnIds(catIds=self.filter_cats, iscrowd=False)
        # filter very small objects
        self.annotations_ids = list(filter(lambda ann_id: thing_coco_api.loadAnns([ann_id])[0]['bbox'][2] >= 50 and
                                                  thing_coco_api.loadAnns([ann_id])[0]['bbox'][3] >= 50,
                                   self.annotations_ids))
        anns_pos = np.random.randint(0, len(self.annotations_ids), 60_000)
        self.annotations_ids = np.take(self.annotations_ids, anns_pos)

    def __len__(self):
        return len(self.annotations_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids = np.take(self.annotations_ids, idx)
        return ids

    def get_img(self, img_id):
        img_file = thing_coco_api.loadImgs([img_id])[0]['file_name']
        img = Image.open(self.image_dir + "/" + img_file).convert("RGB")
        return np.array(img) / 255.0

    def get_img_by_obj(self, obj):
        ann = thing_coco_api.loadAnns([obj])[0]
        img_id = ann["image_id"]
        return self.get_img(img_id)

    def get_trimap_by_id(self, id):
        ann = thing_coco_api.loadAnns([id])[0]
        data = thing_coco_api.annToMask(ann)
        if data.size == 0:
            data = np.array([[1]])
        return data

    def get_rand_objects(self, cnt):
        ids = np.random.randint(0, len(self.annotations_ids), cnt)
        ids = np.take(self.annotations_ids, ids)
        return ids
        #objects = self.get_objects_by_ids(ids)
        #return self.pad_objects(objects)

    def get_cat_num(self):
        return self.cat_num

    def crop_by_bbox(self, mask, bbox):
        x1, y1, x2, y2 = bbox
        x1 = floor(x1)
        y1 = floor(y1)
        x2 = ceil(x2)
        y2 = ceil(y2)
        return mask[y1:y2 + 1, x1:x2 + 1]

    def get_bbox_by_id(self, id):
        ann = thing_coco_api.loadAnns([id])[0]
        x1, y1, w, h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def get_objects_by_ids(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        objects = []
        anns = thing_coco_api.loadAnns(ids)
        for i in range(len(ids)):
            data = thing_coco_api.annToMask(anns[i])
            data = self.crop_by_bbox(data, self.get_bbox_by_id(ids[i]))
            cat_id = anns[i]["category_id"]
            if data.size == 0:
                data = np.array([[cat_id]])
            data[data > 0] = cat_id
            objects.append(data)
        return objects

    def make_square(self, im, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return T.Resize(256).forward(new_im)

    def pad_objects(self, objects, make_copy=False):
        """
        Pads objects with zeros to be 256*256 size (object in the center)
        """
        padded_objects = objects
        if make_copy:
            padded_objects = copy.deepcopy(objects)
        for i in range(len(padded_objects)):
            while padded_objects[i].shape[0] > 256 or padded_objects[i].shape[1] > 256:
                padded_objects[i] = padded_objects[i][::2, ::2]
            while padded_objects[i].shape[0] * 2 < 256 and padded_objects[i].shape[1] * 2 < 256:
                padded_objects[i] = padded_objects[i].repeat(2, axis=0).repeat(2, axis=1)
            h, w = padded_objects[i].shape
            y1 = (256 - h + 1) // 2
            y2 = (256 - h) // 2
            x1 = (256 - w + 1) // 2
            x2 = (256 - w) // 2
            padded_objects[i] = np.pad(padded_objects[i], ((y1, y2), (x1, x2)), 'constant', constant_values=0)
        return np.asarray(padded_objects)


class PeopleDataset:
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.files = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]

    def get_rand_objects(self, cnt):
        ids = np.random.randint(0, len(self.files), cnt)
        files = np.take(self.files, ids)
        images = []
        for img_file in files:
            img = self.get_img(img_file)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)
        return images

    def get_img(self, img_file):
        img = Image.open(img_file).convert("RGBA")
        # img = T.Resize(256).forward(img)
        # threshold = 252
        # arr = np.array(np.asarray(img))
        # r, g, b, a = np.rollaxis(arr, axis=-1)
        # mask = ((r >= threshold) & (g >= threshold) & (b >= threshold))
        # a[mask] = 0
        # plt.imshow(a)
        # plt.show()
        # a = cv.erode(a, (5, 5), iterations=10)
        # #a = cv.GaussianBlur(a, (15, 15), 0)
        # arr = np.dstack((r, g, b, a))
        # img = Image.fromarray(arr, mode='RGBA')
        return img


# dataset = BackgroundDataset(filter_cats=[1], image_dir='coco/images/train2017', transform=T.Compose([T.Resize(256), T.RandomCrop(256), T.ToTensor()]))
# segms = dataset.__getitem__(100)
#
# dataset = PeopleDataset(image_dir='coco/images/objects', transform=T.Compose([T.Resize(256), T.ToTensor()]))
# images = dataset.get_rand_objects(4)
#
# theta = torch.FloatTensor([[[0.1, 0, 0], [0, 0.1, 0]], [[0.25, 0, 0], [0, 0.25, 0]], [[0.5, 0, 0], [0, 0.5, 0]], [[0.75, 0, 0], [0, 0.75, 0]]])
# theta = theta ** torch.FloatTensor([[-1, 1, 1], [1, -1, 1]])
# print(theta)
# grid = F.affine_grid(theta, images.size())
# images = F.grid_sample(images, grid)
#
# color, mask = images[:, :3, :, :], images[:, 3:, :, :]
# fake = color * mask + segms * (1 - mask)
# for i in range(4):
#     plt.imshow(fake[i].permute(1, 2, 0))
#     plt.show()

# dataset = ObjectsDataset(filter_cats=[1], image_dir='coco/images/train2017')
# ids = dataset.get_rand_objects(10000)
#
# for id in ids:
#     img = dataset.get_img_by_obj(id)
#     co = cutout(img, dataset.get_trimap_by_id(id), dataset.get_bbox_by_id(id))
#
#     r, g, b, a = np.rollaxis(co, axis=-1)
#     a = cv.erode(a, (3, 3), iterations=3)
#     a = cv.GaussianBlur(a, (5, 5), 0)
#     co = np.dstack((r, g, b, a))
#
#     result = Image.fromarray((co * 255).astype('uint8'), 'RGBA')
#     result = dataset.make_square(result)
#     result.save("coco/images/objects/" + str(id) + '.png')
