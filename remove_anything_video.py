import torch
import sys
import argparse
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from sam_segment import build_sam_model
from ostrack import build_ostrack_model, get_box_using_ostrack
from pytracking.lib.test.evaluation.data import Sequence
from pytracking.lib.utils.video_utils import video2frames, frames2video
from utils import load_img_to_array, save_array_to_img, dilate_mask

# def setup_args(parser):
#     parser.add_argument(
#         "--input_img", type=str, required=True,
#         help="Path to a single input img",
#     )
#     parser.add_argument(
#         "--coords_type", type=str, required=True,
#         default="key_in", choices=["click", "key_in"], 
#         help="The way to select coords",
#     )
#     parser.add_argument(
#         "--point_coords", type=float, nargs='+', required=True,
#         help="The coordinate of the point prompt, [coord_W coord_H].",
#     )
#     parser.add_argument(
#         "--point_labels", type=int, nargs='+', required=True,
#         help="The labels of the point prompt, 1 or 0.",
#     )
#     parser.add_argument(
#         "--dilate_kernel_size", type=int, default=None,
#         help="Dilate kernel size. Default: None",
#     )
#     parser.add_argument(
#         "--output_dir", type=str, required=True,
#         help="Output path to the directory with results.",
#     )
#     parser.add_argument(
#         "--sam_model_type", type=str,
#         default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
#         help="The type of sam model to load. Default: 'vit_h"
#     )
#     parser.add_argument(
#         "--sam_ckpt", type=str, required=True,
#         help="The path to the SAM checkpoint to use for mask generation.",
#     )
#     parser.add_argument(
#         "--lama_config", type=str,
#         default="./lama/configs/prediction/default.yaml",
#         help="The path to the config file of lama model. "
#              "Default: the config of big-lama",
#     )
#     parser.add_argument(
#         "--lama_ckpt", type=str, required=True,
#         help="The path to the lama checkpoint.",
#     )

class RemoveAnythingVideo(nn.Module):
    def __init__(
            self,
            tracker_target="ostrack",
            tracker_build_args: Dict = None,
            segmentor_target="sam",
            segmentor_build_args: Dict = None,
            inpainter_target="lama",
            inpainter_build_args: Dict = None,
    ):
        super().__init__()
        if tracker_build_args is None:
            tracker_build_args = {
                "tracker_param": "vitb_384_mae_ce_32x4_ep300"
            }
        if segmentor_build_args is None:
            segmentor_build_args = {
                "model_type": "vit_h",
                # "ckpt_p": "./pretrained_models/sam_vit_h_4b8939.pth"
                "ckpt_p": "/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth"
            }
        if inpainter_build_args is None:
            inpainter_build_args = {
                "config_p": "./lama/configs/prediction/default.yaml",
                "ckpt_p": "./pretrained_models/big-lama"
            }
        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args
        )
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args)

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support sam now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="", **kwargs):
        pass

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

    def forward_inpainter(self, img, mask):
        raise NotImplementedError

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

    def visualize_box(self, img, box, save_p="bbox.png"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # x1, y1, x2, y2 = 230, 283, 352, 407
        x1, y1, w, h = box
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(save_p)

    def forward(
            self,
            all_frame_ps: List[str],
            key_frame_idx: int,
            key_frame_point_coords: np.ndarray,
            key_frame_point_labels: np.ndarray,
            key_frame_mask_idx: int = None,
            dilate_kernel_size: int = 15,
    ):
        # get key-frame mask
        assert key_frame_idx == 0, "Only support key frame at the beginning."
        key_frame_p = all_frame_ps[key_frame_idx]
        key_frame = load_img_to_array(key_frame_p)
        key_masks, key_scores = self.forward_segmentor(
            key_frame, key_frame_point_coords, key_frame_point_labels)
        # key-frame mask selection
        if key_frame_mask_idx is not None:
            key_mask = key_masks[key_frame_mask_idx]
        else:
            key_mask = self.mask_selection(key_masks, key_scores)
        key_mask = (key_mask * 255).astype(np.uint8)
        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)

        save_array_to_img(key_mask, "mask.png")

        # get key-frame box
        key_box = self.get_box_from_mask(key_mask)

        # self.visualize_box(key_frame, key_box)
        # raise

        # get all-frame boxes using video tracker
        all_box = self.forward_tracker(all_frame_ps, key_box)

        # get all-frame masks using sam
        all_mask = [key_mask]
        all_frame = [key_frame]
        ref_mask = key_mask
        for frame_p, box in zip(all_frame_ps[1:], all_box[1:]):
            frame = load_img_to_array(frame_p)

            # save_p = f"results/bbox/{Path(frame_p).name}"
            # Path(save_p).parent.mkdir(parents=True, exist_ok=True)
            # self.visualize_box(frame, box, save_p)

            # XYWH -> XYXY
            x, y, w, h = box
            sam_box = np.array([x, y, x + w, y + h])
            masks, scores = self.forward_segmentor(frame, box=sam_box)
            masks = (masks * 255).astype(np.uint8)
            mask = self.mask_selection(masks, scores, ref_mask)
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)
            # mask = self.mask_selection(masks, scores)

            ref_mask = mask
            all_mask.append(mask)
            all_frame.append(frame)

        # get all-frame inpainted results
        # all_frame = self.inpainter(all_frame, all_mask)
        return all_frame, all_mask, all_box


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # setup_args(parser)
    # args = parser.parse_args(sys.argv[1:])
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    point_labels = np.array([1])
    key_frame_mask_idx = 2

    video_path = '/data1/yutao/projects/Inpaint-Anything/example/remove-anything-video/breakdance-flare/original_video.mp4'
    point_coords = np.array([[450, 252]])
    dilate_kernel_size = 15


    output_dir = Path('./results')
    raw_frame_dir = output_dir / Path(video_path).stem / "raw"
    mask_frame_dir = output_dir / Path(video_path).stem / "mask"
    raw_frame_dir.mkdir(exist_ok=True, parents=True)
    mask_frame_dir.mkdir(exist_ok=True, parents=True)

    if Path(video_path).exists():
        video2frames(video_path, raw_frame_dir)

    model = RemoveAnythingVideo()
    # model = RemoveAnythingVideo(segmentor_build_args=args)
    all_frame_ps = sorted(glob.glob(str(raw_frame_dir / "*.jpg")))
    all_frame_ps = all_frame_ps
    all_frame, all_mask, all_box = model(
        all_frame_ps, 0, point_coords, point_labels, key_frame_mask_idx,
        dilate_kernel_size
    )

    for i, mask in enumerate(all_mask):
        img_name = Path(all_frame_ps[i]).name
        save_array_to_img(mask, mask_frame_dir / img_name)


