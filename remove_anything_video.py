import torch
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from pathlib import Path
from PIL import Image
import os
import sys
import argparse
import tempfile
import imageio
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam_segment import build_sam_model
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from ostrack import build_ostrack_model, get_box_using_ostrack
from sttn_video_inpaint import build_sttn_model, \
    inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask, show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a single input video",
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
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
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
        "--vi_ckpt", type=str, required=True,
        help="The path to video inpainter checkpoint.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=2, required=True,
        help="Which mask in the first frame to determine the inpaint region.",
    )
    parser.add_argument(
        "--fps", type=int, default=25, required=True,
        help="FPS of the input and output videos.",
    )

class RemoveAnythingVideo(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target="ostrack",
            segmentor_target="sam",
            inpainter_target="sttn",
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
            "lama": {
                "lama_config": args.lama_config,
                "lama_ckpt": args.lama_ckpt
            },
            "sttn": {
                "model_type": "sttn",
                "ckpt_p": args.vi_ckpt
            }
        }

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args)
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support sam now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="sttn", **kwargs):
        if target == "lama":
            return build_lama_model(**kwargs)
        elif target == "sttn":
            return build_sttn_model(**kwargs)
        else:
            raise NotImplementedError("Only support lama and sttn")

    def forward_tracker(self, frames_ps, init_box):
        init_box = np.array(init_box).astype(np.float32).reshape(-1, 4)
        seq = Sequence("tmp", frames_ps, 'inpaint-anything', init_box)
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

    def forward_inpainter(self, frames, masks):
        if self.inpainter_target == "lama":
            for idx in range(len(frames)):
                frames[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target == "sttn":
            frames = [Image.fromarray(frame) for frame in frames]
            masks = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            frames = inpaint_video_with_builded_sttn(
                self.inpainter, frames, masks, device=self.device)
        else:
            raise NotImplementedError
        return frames

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
            frame_ps: List[str],
            key_frame_idx: int,
            key_frame_point_coords: np.ndarray,
            key_frame_point_labels: np.ndarray,
            key_frame_mask_idx: int = None,
            dilate_kernel_size: int = 15,
    ):
        """
        Mask is 0-1 ndarray in default
        Frame is 0-255 ndarray in default
        """
        assert key_frame_idx == 0, "Only support key frame at the beginning."

        # get key-frame mask
        key_frame_p = frame_ps[key_frame_idx]
        key_frame = iio.imread(key_frame_p)
        key_masks, key_scores = self.forward_segmentor(
            key_frame, key_frame_point_coords, key_frame_point_labels)

        # key-frame mask selection
        if key_frame_mask_idx is not None:
            key_mask = key_masks[key_frame_mask_idx]
        else:
            key_mask = self.mask_selection(key_masks, key_scores)
        
        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)

        # get key-frame box
        key_box = self.get_box_from_mask(key_mask)

        # get all-frame boxes using video tracker
        print("Tracking ...")
        all_box = self.forward_tracker(frame_ps, key_box)

        # get all-frame masks using sam
        print("Segmenting ...")
        all_mask = [key_mask]
        all_frame = [key_frame]
        ref_mask = key_mask
        for frame_p, box in zip(frame_ps[1:], all_box[1:]):
            frame = iio.imread(frame_p)

            # XYWH -> XYXY
            x, y, w, h = box
            sam_box = np.array([x, y, x + w, y + h])
            masks, scores = self.forward_segmentor(frame, box=sam_box)
            mask = self.mask_selection(masks, scores, ref_mask)
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)

            ref_mask = mask
            all_mask.append(mask)
            all_frame.append(frame)

        # get all-frame inpainted results
        print("Inpainting ...")
        all_frame = self.forward_inpainter(all_frame, all_mask)
        return all_frame, all_mask, all_box


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
    python remove_anything_video.py \
        --input_video ./example/video/paragliding/original_video.mp4 \
        --coords_type key_in \
        --point_coords 652 162 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama \
        --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
        --vi_ckpt ./pretrained_models/sttn.pth \
        --mask_idx 2 \
        --fps 25
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import logging
    logger = logging.getLogger('imageio')
    logger.setLevel(logging.ERROR)

    dilate_kernel_size = args.dilate_kernel_size
    key_frame_mask_idx = args.mask_idx
    video_raw_p = args.input_video
    frame_raw_glob = None
    fps = args.fps
    num_frames = 10000
    output_dir = args.output_dir
    output_dir = Path(f"{output_dir}")
    frame_mask_dir = output_dir / f"mask_{dilate_kernel_size}"
    video_mask_p = output_dir / f"mask_{dilate_kernel_size}.mp4"
    video_rm_w_mask_p = output_dir / f"removed_w_mask_{dilate_kernel_size}.mp4"
    video_w_mask_p = output_dir / f"w_mask_{dilate_kernel_size}.mp4"
    video_w_box_p = output_dir / f"w_box_{dilate_kernel_size}.mp4"
    frame_mask_dir.mkdir(exist_ok=True, parents=True)

    # load raw video or raw frames
    if Path(video_raw_p).exists():
        all_frame = iio.mimread(video_raw_p)
        fps = imageio.v3.immeta(video_raw_p, exclude_applied=False)["fps"]

        # tmp frames
        frame_ps = []
        for i in range(len(all_frame)):
            frame_p = str(mkstemp(suffix=f"{i:0>6}.png"))
            frame_ps.append(frame_p)
            iio.imwrite(frame_ps[i], all_frame[i])
    else:
        assert frame_raw_glob is not None
        frame_ps = sorted(glob.glob(frame_raw_glob))
        all_frame = [iio.imread(frame_p) for frame_p in frame_ps]
        fps = 25
        # save tmp video
        iio.mimwrite(video_raw_p, all_frame, fps=fps)

    frame_ps = frame_ps[:num_frames]
    
    point_labels = np.array(args.point_labels)
    if args.coords_type == "click":
        point_coords = get_clicked_point(frame_ps[0])
    elif args.coords_type == "key_in":
        point_coords = args.point_coords
    point_coords = np.array([point_coords])

    # inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RemoveAnythingVideo(args)
    model.to(device)
    with torch.no_grad():
        all_frame_rm_w_mask, all_mask, all_box = model(
            frame_ps, 0, point_coords, point_labels, key_frame_mask_idx,
            dilate_kernel_size
        )
    # visual removed results
    iio.mimwrite(video_rm_w_mask_p, all_frame_rm_w_mask, fps=fps)

    # visual mask
    all_mask = [np.uint8(mask * 255) for mask in all_mask]
    for i in range(len(all_mask)):
        mask_p = frame_mask_dir /  f"{i:0>6}.jpg"
        iio.imwrite(mask_p, all_mask[i])
    iio.mimwrite(video_mask_p, all_mask, fps=fps)
    # visual video with mask
    tmp = []
    for i in range(len(all_mask)):
        tmp.append(show_img_with_mask(all_frame[i], all_mask[i]))
    iio.mimwrite(video_w_mask_p, tmp, fps=fps)
    tmp = []
    # visual video with box
    for i in range(len(all_box)):
        tmp.append(show_img_with_box(all_frame[i], all_box[i]))
    iio.mimwrite(video_w_box_p, tmp, fps=fps)
