import sys
import numpy as np
from pymatting import *
from math import ceil, floor, sqrt
import cv2


def get_margin(margin, om):
    if margin < 1:
        return round(sqrt(100 * margin * np.count_nonzero(om)) / 100)
    return int(margin)


def cutout_object(image, trimap, alpha_matting):
    # estimate alpha from image and trimap
    if alpha_matting == "cf":
        alpha = estimate_alpha_cf(image, trimap)
    elif alpha_matting == "rw":
        alpha = estimate_alpha_rw(image, trimap)
    elif alpha_matting == "knn":
        alpha = estimate_alpha_knn(image, trimap)
    elif alpha_matting == "lkm":
        alpha = estimate_alpha_lkm(image, trimap)
    elif alpha_matting == "lbdm":
        alpha = estimate_alpha_lbdm(image, trimap)
    else:
        raise Exception("No such alpha matting algorithm")
    # estimate foreground from image and alpha
    alpha[alpha < 0.6] = 0
    foreground = estimate_foreground_ml(image, alpha)
    cutout = stack_images(foreground, alpha)
    return cutout


def crop_by_bbox(img, mask, bbox, margin):
    x1, y1, x2, y2 = bbox
    h, w = mask.shape
    x1 = max(0, floor(x1) - margin)
    y1 = max(0, floor(y1) - margin)
    x2 = min(ceil(x2) + margin, w - 1)
    y2 = min(ceil(y2) + margin, h - 1)
    return img[y1:y2 + 1, x1:x2 + 1], mask[y1:y2 + 1, x1:x2 + 1]


def mask_to_trimap(mask, margin):
    kernel = np.ones((3, 3), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=margin)
    mask_dilated = cv2.dilate(mask, kernel, iterations=margin)
    trimap = mask_dilated
    trimap[trimap > 0] = 0.5
    trimap[mask_eroded > 0] = 1
    return trimap


def cutout(img, mask, bbox=None, margin=0.22, alpha_matting="cf"):
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    mask = mask.astype(float)
    margin = get_margin(margin, mask)
    if bbox is not None:
        img, mask = crop_by_bbox(img, mask, bbox, margin)
    trimap = mask_to_trimap(mask, margin)
    # cutout object
    cutout = cutout_object(img, trimap, alpha_matting)
    return cutout
