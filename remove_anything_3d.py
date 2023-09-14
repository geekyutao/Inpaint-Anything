import torch
import sys
import os
import cv2
import argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import glob
import imageio.v2 as iio
import tempfile
import matplotlib.patches as patches
from typing import Any, Dict, List
from pytracking.lib.test.evaluation.data import Sequence
from sam_segment import build_sam_model
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from ostrack import build_ostrack_model, get_box_using_ostrack
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from nerf.run_nerf import train


def setup_args(parser):
    #remove object from source images option
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the directory with source images",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, default=1, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
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
    parser.add_argument(
        "--tracker_ckpt", type=str, required=True,
        help="The path to tracker checkpoint.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=1, required=True,
        help="Which mask in the first frame to determine the inpaint region.",
    )

    #novel views synthesis option
    parser.add_argument(
        '--config', type=str, default=None,
        help='config file path'
        )
    parser.add_argument(
        "--expname", type=str, 
        help='experiment name'
        )
    
    # training options
    parser.add_argument(
        "--netdepth", type=int, default=8, 
        help='layers in network'
        )
    parser.add_argument(
        "--netwidth", type=int, default=256, 
        help='channels per layer'
        )
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, 
        help='layers in fine network'
        )
    parser.add_argument(
        "--netwidth_fine", type=int, default=256, 
        help='channels per layer in fine network'
        )
    parser.add_argument(
        "--N_rand", type=int, default=32*32*4, 
        help='batch size (number of random rays per gradient step)'
        )
    parser.add_argument(
        "--lrate", type=float, default=5e-4, 
        help='learning rate'
        )
    parser.add_argument(
        "--lrate_decay", type=int, default=250, 
        help='exponential learning rate decay (in 1000 steps)'
        )
    parser.add_argument(
        "--chunk", type=int, default=1024*32, 
         help='number of rays processed in parallel, decrease if running out of memory'
         )
    parser.add_argument(
        "--netchunk", type=int, default=1024*64, 
        help='number of pts sent through network in parallel, decrease if running out of memory'
        )
    parser.add_argument(
        "--no_batching", action='store_true', 
        help='only take random rays from 1 image at a time'
        )
    parser.add_argument(
        "--no_reload", action='store_true', 
        help='do not reload weights from saved ckpt'
        )
    parser.add_argument(
        "--ft_path", type=str, default=None, 
        help='specific weights npy file to reload for coarse network'
        )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, 
        help='number of coarse samples per ray'
        )
    parser.add_argument(
        "--N_importance", type=int, default=64,
        help='number of additional fine samples per ray'
        )
    parser.add_argument(
        "--perturb", type=float, default=1.,
        help='set to 0. for no jitter, 1. for jitter'
        )
    parser.add_argument(
        "--use_viewdirs", action='store_true', 
        help='use full 5D input instead of 3D'
        )
    parser.add_argument(
        "--i_embed", type=int, default=0, 
        help='set 0 for default positional encoding, -1 for none'
        )
    parser.add_argument(
        "--multires", type=int, default=10, 
        help='log2 of max freq for positional encoding (3D location)'
        )
    parser.add_argument(
        "--multires_views", type=int, default=4, 
        help='log2 of max freq for positional encoding (2D direction)'
        )
    parser.add_argument(
        "--raw_noise_std", type=float, default=0., 
        help='std dev of noise added to regularize sigma_a output, 1e0 recommended'
        )

    parser.add_argument(
        "--render_only", action='store_true', 
        help='do not optimize, reload weights and render out render_poses path'
        )
    parser.add_argument(
        "--render_test", action='store_true', 
        help='render the test set instead of render_poses path'
        )
    parser.add_argument(
        "--render_factor", type=int, default=0, 
        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview'
        )

    # training options
    parser.add_argument(
        "--precrop_iters", type=int, default=0,
        help='number of steps to train on central crops'
        )
    parser.add_argument(
        "--precrop_frac", type=float,
        default=.5, help='fraction of img taken for central crops'
        ) 

    # dataset options
    parser.add_argument(
        "--dataset_type", type=str, default='llff', 
        help='options: llff / blender / deepvoxels'
        )
    parser.add_argument(
        "--testskip", type=int, default=8, 
        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels'
        )

    ## deepvoxels flags
    parser.add_argument(
        "--shape", type=str, default='greek', 
        help='options : armchair / cube / greek / vase'
        )

    ## blender flags
    parser.add_argument(
        "--white_bkgd", action='store_true', 
        help='set to render synthetic data on a white bkgd (always use for dvoxels)'
        )
    parser.add_argument(
        "--half_res", action='store_true', 
        help='load blender synthetic data at 400x400 instead of 800x800'
        )

    ## llff flags
    parser.add_argument(
        "--factor", type=int, default=4, 
        help='downsample factor for LLFF images'
        )
    parser.add_argument(
        "--no_ndc", action='store_true', 
        help='do not use normalized device coordinates (set for non-forward facing scenes)'
        )
    parser.add_argument(
        "--lindisp", action='store_true', 
        help='sampling linearly in disparity rather than depth'
        )
    parser.add_argument(
        "--spherify", action='store_true', 
        help='set for spherical 360 scenes'
        )
    parser.add_argument(
        "--llffhold", type=int, default=8, 
        help='will take every 1/N images as LLFF test set, paper uses 8'
        )

    # logging/saving options
    parser.add_argument(
        "--i_print",   type=int, default=100, 
        help='frequency of console printout and metric loggin'
        )
    parser.add_argument(
        "--i_img",     type=int, default=500, 
        help='frequency of tensorboard image logging'
        )
    parser.add_argument(
        "--i_weights", type=int, default=10000, 
        help='frequency of weight ckpt saving'
        )
    parser.add_argument(
        "--i_testset", type=int, default=50000, 
        help='frequency of testset saving'
        )
    parser.add_argument(
        "--i_video",   type=int, default=50000, 
        help='frequency of render_poses video saving'
        )

class RemoveAnything3D(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target="ostrack",
            segmentor_target="sam",
            inpainter_target="lama",
    ):
        super().__init__()
        tracker_build_args = {
            "tracker_param": args.tracker_ckpt
        }
        segmentor_build_args = {
            "model_type": args.sam_model_type,
            "ckpt_p": args.sam_ckpt
        }
        inpainter_build_args = {
                "config_p": args.lama_config,
                "ckpt_p": args.lama_ckpt
        }

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args)
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args)
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support sam now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="lama", **kwargs):
        assert  target == "lama", "Only support lama now."
        return build_lama_model(**kwargs)


    def forward_tracker(self, image_ps, init_box):
        init_box = np.array(init_box).astype(np.float32).reshape(-1, 4)
        seq = Sequence("tmp", image_ps, 'inpaint-anything', init_box)
        all_box_xywh = get_box_using_ostrack(self.tracker, seq)
        return all_box_xywh

    def forward_segmentor(self, img, point_coords=None, point_labels=None,
                          box=None, mask_input=None, multimask_output=True,
                          return_logits=False):
        self.segmentor.set_image(img)

        masks, scores, logits = self.segmentor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, images, masks):
        if self.inpainter_target == "lama":
            for idx in range(len(images)):
                images[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, images[idx], masks[idx], device=self.device)
        else:
            raise NotImplementedError
        return images

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def mask_selection(self, masks, scores, ref_mask=None, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            if ref_mask is not None:
                mse = np.mean(
                    (masks.astype(np.int32) - ref_mask.astype(np.int32))**2,
                    axis=(-2, -1)
                )
                idx = mse.argmin()
            else:
                idx = scores.argmax()
            return masks[idx]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return np.array([x, y, w, h])

    def forward(
            self,
            image_ps: List[str],
            key_image_idx: int,
            key_image_point_coords: np.ndarray,
            key_image_point_labels: np.ndarray,
            key_image_mask_idx: int = None,
            dilate_kernel_size: int = 15,
    ):
        """
        Mask is 0-1 ndarray in default
        """
        assert key_image_idx == 0, "Only support key image at the beginning."

        # get key-image mask
        key_image_p = image_ps[key_image_idx]
        key_image = iio.imread(key_image_p)
        key_masks, key_scores = self.forward_segmentor(
            key_image, key_image_point_coords, key_image_point_labels)

        # key-image mask selection
        if key_image_mask_idx is not None:
            key_mask = key_masks[key_image_mask_idx]
        else:
            key_mask = self.mask_selection(key_masks, key_scores)
        
        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)

        # get key-image box
        key_box = self.get_box_from_mask(key_mask)

        # get all-image boxes using tracker
        print("Tracking ...")
        all_box = self.forward_tracker(image_ps, key_box)

        # get all-image masks using sam
        print("Segmenting ...")
        all_mask = [key_mask]
        all_image = [key_image]
        ref_mask = key_mask
        for image_p, box in zip(image_ps[1:], all_box[1:]):
            image = iio.imread(image_p)

            # XYWH -> XYXY
            x, y, w, h = box
            sam_box = np.array([x, y, x + w, y + h])
            masks, scores = self.forward_segmentor(image, box=sam_box)
            mask = self.mask_selection(masks, scores, ref_mask)
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)

            ref_mask = mask
            all_mask.append(mask)
            all_image.append(image)

        # get all-image inpainted results
        print("Inpainting ...")
        all_image = self.forward_inpainter(all_image, all_mask)
        return all_image, all_mask, all_box
    
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels,
                size=(width * 0.04) ** 2)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_box(img, box):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(width / dpi / 0.77, height / dpi / 0.77))
    ax.imshow(img)
    ax.axis('off')

    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp(".png")
    fig.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

if __name__ == "__main__":
    """Example usage:
    python remove_anything_3d.py \
        --input_dir ./example/3d/horns \
        --coords_type key_in \
        --point_coords 830 405 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama \
        --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
        --mask_idx 1 \
        --config ./nerf/configs/horns.txt \
        --expname horns
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dilate_kernel_size = args.dilate_kernel_size
    key_image_mask_idx = args.mask_idx
    images_raw_dir = args.input_dir
    factor = args.factor

    images_raw_dir = Path(f"{images_raw_dir}")
    removed_dir = images_raw_dir / f"images_remove_{factor}"
    image_mask_dir = removed_dir / f"mask_{dilate_kernel_size}" #存mask images
    images_rm_w_mask_dir = removed_dir / f"removed_with_mask_{dilate_kernel_size}" #removed images
    images_w_mask_dir = removed_dir / f"w_mask_{dilate_kernel_size}"
    images_w_box_dir = removed_dir / f"w_box_{dilate_kernel_size}"
    removed_dir.mkdir(exist_ok=True, parents=True)
    image_mask_dir.mkdir(exist_ok=True, parents=True)
    images_rm_w_mask_dir.mkdir(exist_ok=True, parents=True)
    images_w_mask_dir.mkdir(exist_ok=True, parents=True)
    images_w_box_dir.mkdir(exist_ok=True, parents=True)


    #load source multi-view images
    image_ps =[]
    assert Path(images_raw_dir).exists()
    if args.factor is not None:
        images_raw_dir = os.path.join(images_raw_dir,'images'+'_{}'.format(factor))
    else:
        images_raw_dir = os.path.join(images_raw_dir,'images')
    image_ps = sorted(glob.glob(os.path.join(images_raw_dir,'*.png')))

    point_labels = np.array(args.point_labels)
    if args.coords_type == "click":
        point_coords = get_clicked_point(image_ps[0])
    elif args.coords_type == "key_in":
        point_coords = args.point_coords
    point_coords = np.array([point_coords])

    #remove object from source images 
    # device = "cuda:4" if torch.cuda.is_available() else "cpu"
    model = RemoveAnything3D(args)
    model.to(device)
    with torch.no_grad():
        all_images_rm_w_mask, all_mask, all_box = model(
            image_ps, 0, point_coords, point_labels, key_image_mask_idx,
            dilate_kernel_size
        )

    #save removed images
    for i in range(len(all_images_rm_w_mask)):
        all_images_rm_w_mask_p = images_rm_w_mask_dir / f"{Path(image_ps[i]).stem}.png"
        # images_raw_p = images_raw_dir / f"{Path(image_ps[i]).stem}.png"
        save_array_to_img(all_images_rm_w_mask[i], all_images_rm_w_mask_p)
        # save_array_to_img(all_images_rm_w_mask[i], images_raw_p)
    #save the mask
    all_mask = [np.uint8(mask * 255) for mask in all_mask]
    for i in range(len(all_mask)):
        all_mask_p = image_mask_dir / f"{Path(image_ps[i]).stem}.png"
        save_array_to_img(all_mask[i], all_mask_p)
    #save the source images with mask
    images_w_mask = []
    for i in range(len(all_mask)):
        images_w_mask.append(show_img_with_mask(iio.imread(image_ps[i]), all_mask[i]))
        images_w_mask_p = images_w_mask_dir / f"{Path(image_ps[i]).stem}.png"
        save_array_to_img(images_w_mask[i], images_w_mask_p)
    #save the source images with box
    images_w_box = []
    for i in range(len(all_box)):
        images_w_box.append(show_img_with_box(iio.imread(image_ps[i]), all_box[i]))
        images_w_box_p = images_w_box_dir / f"{Path(image_ps[i]).stem}.png"
        save_array_to_img(images_w_box[i], images_w_box_p)

    #novel view synthesis with removed images
    train(args)
    

    
