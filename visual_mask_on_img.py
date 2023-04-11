import cv2
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import glob


def load_img_to_array(img_p):
    return np.array(Image.open(img_p))


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def show_mask(ax, mask, random_color=False):
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)


def setup_args(parser):
    parser.add_argument(
        "--img_glob", type=str, required=True,
        help="Glob to the input images",
    )
    parser.add_argument(
        "--mask_suffix_glob", type=str, default="_mask.png",
        help="Suffix of the input mask",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the directory of the masked imgs",
    )

if __name__ == "__main__":
    """
    python visual_mask_on_img.py --img_glob "recmask/*.jpg" --output_dir recmask 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    img_ps = sorted(glob.glob(args.img_glob))
    for img_p in img_ps:
        img = load_img_to_array(img_p)
        img_stem = Path(img_p).stem

        mask_glob_name = f"{img_stem}_{args.mask_suffix}.png"
        mask_glob = str(Path(img_p).with_name(mask_glob_name))
        mask_ps = sorted(glob.glob(mask_glob))

        for mask_p in mask_ps:
            # mask_suffix =
            mask = load_img_to_array(mask_p)
            mask = mask.astype(np.uint8)

            # path to the results
            masked_img_name = f"{img_stem}_masked_{args.mask_suffix}.png"
            masked_img_p = Path(args.output_dir) / masked_img_name

            # save the mask
            save_array_to_img(mask, mask_p)

            # save the masked image
            dpi = plt.rcParams['figure.dpi']
            height, width = img.shape[:2]
            plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
            plt.imshow(img)
            plt.axis('off')
            show_mask(plt.gca(), mask, random_color=False)
            plt.savefig(masked_img_p, bbox_inches='tight', pad_inches=0)
            plt.close()
