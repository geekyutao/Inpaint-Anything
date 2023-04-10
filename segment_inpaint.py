import sys

import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse

from typing import Any, Dict, List
from PIL import Image
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
from .inpaint import rgb_preprocess, inpaint_image_with_lama


def load_image_to_array(image_path):
    return np.array(Image.open(image_path), dtype=np.float32)


def predict_mask_with_sam(
        image: np.ndarray,
        point_coords: List[List[int]],
        point_labels: List[int],
        model_type: str,
        checkpoint: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def show_points(ax, coords: List[List[int]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
               s=size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))

def setup_args(parser):
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--point_coords", type=int, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    image_arr = load_image_to_array(args.input_image)
    image_stem = Path(args.input_image).stem

    masks, scores, logits = predict_mask_with_sam(
        image_arr,
        [args.point_coords],
        args.point_labels,
        checkpoint=args.sam_ckpt,
        model_type=args.sam_model_type,
        device="cuda",
    )

    masks = (masks * 255).astype(np.uint8)
    dilate_factor = 15
    for idx in range(len(masks)):
        plt.figure()
        plt.imshow(image_arr)
        plt.axis('off')
        show_points(plt.gca(), [args.point_coords], args.point_labels)
        mask = masks[idx]
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        mask_image = show_mask(
            (mask / 255).astype(np.bool_), random_color=False)
        plt.imshow(mask_image)
        plt.savefig(
            "example/{}_masked_{}.png".format(image_stem, idx), dpi=300,
            bbox_inches='tight', pad_inches=0
        )

        # save the mask
        Image.fromarray(mask).save("example/{}_mask_{}.png".format(image_stem, idx))
        plt.close()

        # img, mask = rgb_preprocess(img_path, mask_path)

        # cur_res = inpaint_image_with_lama(img, mask, config_path, ckpt_path)