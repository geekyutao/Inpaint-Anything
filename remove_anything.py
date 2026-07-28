import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import select_masks, ALL_MODEL_TYPES, DEFAULT_SAM3_CKPT
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', default=None,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', default=None,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--text_select", type=str, default=None,
        help="SAM 3 only: pick the target with a noun phrase (e.g. 'the dog') "
             "instead of a point. All matching instances are removed together.",
    )
    parser.add_argument(
        "--text_confidence", type=float, default=0.5,
        help="Confidence threshold for --text_select. Default: 0.5",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="sam3", choices=list(ALL_MODEL_TYPES),
        help="The type of sam model to load. Default: 'sam3'",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, default=DEFAULT_SAM3_CKPT,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--inpaint_model", type=str, default="lama",
        help="What fills the hole: 'lama' (fast, the default), 'flux' "
             "(FLUX.1-Fill, best quality, ~20GB VRAM), 'sdxl', or any "
             "diffusers inpainting model id.",
    )
    parser.add_argument(
        "--remove_prompt", type=str, default="",
        help="Prompt used by the diffusion models when removing. Empty means "
             "'just continue the background'. Ignored by lama.",
    )


if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img ./example/remove-anything/dog.jpg \
        --coords_type key_in \
        --point_coords 200 450 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "sam3" \
        --sam_ckpt ./pretrained_models/sam3.pt \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama

    Or pick the target with a text phrase instead of a point (SAM 3 only):
    python remove_anything.py \
        --input_img ./example/remove-anything/dog.jpg \
        --text_select "dog" \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --lama_ckpt ./pretrained_models/big-lama
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latest_coords = None
    if not args.text_select:
        if args.coords_type == "click":
            latest_coords = get_clicked_point(args.input_img)
        elif args.coords_type == "key_in":
            latest_coords = args.point_coords
    img = load_img_to_array(args.input_img)

    masks = select_masks(
        img,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
        point_coords=latest_coords,
        point_labels=args.point_labels,
        text_select=args.text_select,
        text_confidence=args.text_confidence,
    )

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        if latest_coords is not None:
            show_points(plt.gca(), [latest_coords], args.point_labels,
                        size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # inpaint the masked image
    use_lama = args.inpaint_model == "lama"
    if use_lama:
        lama = build_lama_model(args.lama_config, args.lama_ckpt, device=device)
    else:
        from stable_diffusion_inpaint import fill_img_with_sd

    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        if use_lama:
            img_inpainted = inpaint_img_with_builded_lama(
                lama, img, mask, device=device)
        else:
            img_inpainted = fill_img_with_sd(
                img, mask, args.remove_prompt, device=device,
                model_id=args.inpaint_model)
        save_array_to_img(img_inpainted, img_inpainted_p)
