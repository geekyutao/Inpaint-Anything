import cv2
import numpy as np
from PIL import Image


def load_img_to_array(img_p):
    return np.array(Image.open(img_p))


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    if np.max(mask) == 1:
        mask = (mask * 255).astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    mask = (mask / 255).astype(np.bool_)
    return mask