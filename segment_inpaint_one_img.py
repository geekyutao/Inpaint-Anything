import cv2
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List

from segment_anything import SamPredictor, sam_model_registry
from inpaint import inpaint_img_with_lama


def load_img_to_array(img_p):
    return np.array(Image.open(img_p))


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[int]],
        point_labels: List[int],
        model_type: str,
        ckpt_p: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def show_mask(ax, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[int]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)


def dilate_mask(mask, dilate_factor=15):
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    mask = (mask / 255).astype(np.bool_)
    return mask


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
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

    img = load_img_to_array(args.input_img)
    img_stem = Path(args.input_img).stem

    masks, _, _ = predict_masks_with_sam(
        img,
        [args.point_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device="cuda",
    )
    masks = masks.astype(np.uint8)

    # dilate mask to avoid unmasked edge effect
    dilate_factor = 15
    masks = [dilate_mask(mask, dilate_factor) for mask in masks]

    # visualize the segmentation results
    for idx, mask in enumerate(masks):
        # path to the results
        pointed_img_p = Path(args.output_dir) / f"{img_stem}_pointed.png"
        masked_img_p = Path(args.output_dir) / f"{img_stem}_masked_{idx}.png"
        mask_p = Path(args.output_dir) / f"{img_stem}_mask_{idx}.png"

        # save the mask
        save_array_to_img(mask*255, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(pointed_img_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(masked_img_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        inpainted_img_p = Path(args.output_dir) / f"{img_stem}_inpainted_{idx}.png"
        inpainted_img = inpaint_img_with_lama(
            img, mask*255, args.lama_config, args.lama_ckpt)
        save_array_to_img(inpainted_img, inpainted_img_p)