import torch
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from pathlib import Path
from PIL import Image
import os
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
from utils import dilate_mask, show_mask, show_points


class RemoveAnythingVideo(nn.Module):
    build_args = {
        "ostrack": {
            "tracker_param": "vitb_384_mae_ce_32x4_ep300"
        },
        "sam": {
            "model_type": "vit_h",
            "ckpt_p": "./pretrained_models/sam_vit_h_4b8939.pth"
        },
        "lama": {
            "lama_config": "./lama/configs/prediction/default.yaml",
            "lama_ckpt": "./pretrained_models/big-lama"
        },
        "sttn": {
            "model_type": "sttn",
            "ckpt_p": "./pretrained_models/sttn.pth"
        }
    }
    def __init__(
            self,
            tracker_target="ostrack",
            tracker_build_args: Dict = None,
            segmentor_target="sam",
            segmentor_build_args: Dict = None,
            inpainter_target="sttn",
            inpainter_build_args: Dict = None,
    ):
        super().__init__()
        if tracker_build_args is None:
            tracker_build_args = self.build_args[tracker_target]
        if segmentor_build_args is None:
            segmentor_build_args = self.build_args[segmentor_target]
        if inpainter_build_args is None:
            inpainter_build_args = self.build_args[inpainter_target]

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

        # tmp_dir = Path("results/tmp")
        # tmp_dir.mkdir(exist_ok=True, parents=True)
        # iio.imwrite(tmp_dir / "img.png", key_frame)
        # iio.imwrite(tmp_dir / "img_mask.png", np.uint8(key_mask * 255))
        # iio.imwrite(tmp_dir / "img_w_mask.png", show_img_with_mask(
        #     key_frame, key_mask))
        # iio.imwrite(tmp_dir / "img_w_point.png", show_img_with_point(
        #     key_frame, key_frame_point_coords, key_frame_point_labels))
        # iio.imwrite(tmp_dir / "img_w_box.png",
        #     show_img_with_box(key_frame, key_box))

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
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)


@torch.no_grad()
def main():
    import logging
    logger = logging.getLogger('imageio')
    logger.setLevel(logging.ERROR)

    # video_raw_p = './results/baymax.mp4'
    # point_coords = np.array([[868, 813]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 50
    # video_raw_p = './results/blackswan.mp4'
    # point_coords = np.array([[329, 315]])
    # key_frame_mask_idx = 1
    # dilate_kernel_size = 50
    # video_raw_p = './results/bmx-trees.mp4'
    # point_coords = np.array([[448, 205]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 15
    # video_raw_p = './results/boat.mp4'
    # point_coords = np.array([[405, 263]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 15
    # video_raw_p = './results/breakdance-flare.mp4'
    # point_coords = np.array([[450, 252]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 15
    # video_raw_p = './results/car-turn.mp4'
    # point_coords = np.array([[744, 264]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 35
    # video_raw_p = './results/dance_p1.mp4'
    # point_coords = np.array([[421, 765]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 50
    # video_raw_p = './results/ikun.mp4'
    # point_coords = np.array([[290, 341]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 15
    # video_raw_p = './results/lalaland.mp4'
    # point_coords = np.array([[846, 475]])
    # key_frame_mask_idx = 2
    # dilate_kernel_size = 50
    video_raw_p = './results/tennis.mp4'
    frame_raw_glob = None
    point_coords = np.array([[374, 209]])
    key_frame_mask_idx = 2
    dilate_kernel_size = 20

    point_labels = np.array([1])
    num_frames = 10000

    # pre-defined save path
    output_dir = Path('./results')
    video_stem = Path(video_raw_p).stem
    frame_mask_dir = output_dir / video_stem / "mask"
    video_mask_p = output_dir / video_stem / "mask.mp4"
    video_rm_w_mask_p = output_dir / video_stem / f"removed_w_mask.mp4"
    video_w_mask_p = output_dir / video_stem / f"w_mask.mp4"
    video_w_box_p = output_dir / video_stem / f"w_box.mp4"
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

    # inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RemoveAnythingVideo()
    model.to(device)
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


if __name__ == "__main__":
    main()