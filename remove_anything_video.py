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
from sam_segment import build_sam_model, ALL_MODEL_TYPES, DEFAULT_SAM3_CKPT
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from sam3_video_track import build_sam3_video_tracker, track_masks_in_video


def _import_ostrack():
    """OSTrack pulls in the pytracking tree and mutates sys.path, so it is only
    imported when that backend is actually selected."""
    from ostrack import build_ostrack_model, get_box_using_ostrack
    from pytracking.lib.test.evaluation.data import Sequence
    return build_ostrack_model, get_box_using_ostrack, Sequence
from sttn_video_inpaint import build_sttn_model, \
    inpaint_video_with_builded_sttn
from propainter_video_inpaint import build_propainter_model, \
    inpaint_video_with_builded_propainter, DEFAULT_PROPAINTER_CKPT_DIR
from utils import dilate_mask, show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a single input video",
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
        help="SAM 3 only: pick the target in the first frame with a noun "
             "phrase instead of a point. The highest-scoring instance is "
             "tracked through the video.",
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
        help="The path to the lama checkpoint. Only used with --vi_model lama.",
    )
    parser.add_argument(
        "--tracker", type=str, default="sam3", choices=["sam3", "ostrack"],
        help="How to follow the object across frames. 'sam3' propagates the "
             "mask with SAM 3's video predictor and needs no extra checkpoint; "
             "'ostrack' is the legacy box tracker. Default: sam3",
    )
    parser.add_argument(
        "--tracker_ckpt", type=str, default=None,
        help="OSTrack config name, resolved to "
             "./pytracking/pretrain/<name>.pth. Only used with "
             "--tracker ostrack.",
    )
    parser.add_argument(
        "--vi_model", type=str, default="propainter",
        choices=["propainter", "sttn", "lama"],
        help="Video inpainting backend. 'propainter' (ICCV 2023) is the "
             "strongest, 'sttn' is the legacy one, 'lama' inpaints frame by "
             "frame with no temporal consistency. Default: propainter",
    )
    parser.add_argument(
        "--vi_ckpt", type=str, default=None,
        help="Checkpoint for the video inpainter. Defaults to "
             f"'{DEFAULT_PROPAINTER_CKPT_DIR}' for propainter (downloaded on "
             "first use) and './pretrained_models/sttn.pth' for sttn.",
    )
    parser.add_argument(
        "--vi_fp16", action="store_true",
        help="Run ProPainter in half precision to cut GPU memory use.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=None,
        help="Which of SAM's three first-frame candidates to track (0, 1 or "
             "2). Left unset, the sam3 tracker segments the first frame "
             "itself and OSTrack picks the highest-scoring candidate.",
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Fallback FPS, used only when the input video carries no FPS "
             "metadata.",
    )

class RemoveAnythingVideo(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target=None,
            segmentor_target="sam",
            inpainter_target=None,
    ):
        super().__init__()
        tracker_target = tracker_target or getattr(args, "tracker", "sam3")
        inpainter_target = inpainter_target or getattr(args, "vi_model", "propainter")
        self.vi_fp16 = getattr(args, "vi_fp16", False)
        tracker_build_args = {
            "ostrack": {"tracker_param": args.tracker_ckpt},
            "sam3": {"ckpt_p": args.sam_ckpt, "device": self.device},
        }
        segmentor_build_args = {
            "model_type": args.sam_model_type,
            "ckpt_p": args.sam_ckpt
        }
        inpainter_build_args = {
            "lama": {
                "config_p": args.lama_config,
                "ckpt_p": args.lama_ckpt
            },
            "sttn": {
                "model_type": "sttn",
                "ckpt_p": args.vi_ckpt or "./pretrained_models/sttn.pth"
            },
            "propainter": {
                "ckpt_p": args.vi_ckpt or DEFAULT_PROPAINTER_CKPT_DIR
            },
        }

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args[tracker_target])
        # SAM 3's video tracker segments the first frame from a point on its own,
        # and for text prompts it exposes the detector it already holds. A
        # separate image-level model is therefore only needed on the OSTrack
        # path, which drives SAM frame by frame.
        if tracker_target == "sam3":
            self.segmentor = getattr(self.tracker, "text_segmentor", None)
        else:
            self.segmentor = self.build_segmentor(
                segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        if target == "sam3":
            return build_sam3_video_tracker(**kwargs)
        elif target == "ostrack":
            if not kwargs.get("tracker_param"):
                raise ValueError(
                    "--tracker ostrack requires --tracker_ckpt, e.g. "
                    "vitb_384_mae_ce_32x4_ep300")
            build_ostrack_model, _, _ = _import_ostrack()
            return build_ostrack_model(**kwargs)
        raise NotImplementedError("Only sam3 and ostrack are supported")

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="propainter", **kwargs):
        if target == "lama":
            return build_lama_model(**kwargs)
        elif target == "sttn":
            return build_sttn_model(**kwargs)
        elif target == "propainter":
            return build_propainter_model(**kwargs)
        else:
            raise NotImplementedError(
                "Only lama, sttn and propainter are supported")

    def forward_tracker(self, frames_ps, init_box):
        _, get_box_using_ostrack, Sequence = _import_ostrack()
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

    def forward_segmentor_text(self, img, text, confidence=0.5):
        if not hasattr(self.segmentor, "predict_text"):
            raise ValueError(
                "--text_select requires --sam_model_type sam3."
            )
        self.segmentor.set_image(img)
        masks, scores, _ = self.segmentor.predict_text(
            text, confidence_threshold=confidence)
        self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, frames, masks):
        if self.inpainter_target == "lama":
            for idx in range(len(frames)):
                frames[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target in ("sttn", "propainter"):
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_masks = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            if self.inpainter_target == "sttn":
                frames = inpaint_video_with_builded_sttn(
                    self.inpainter, pil_frames, pil_masks, device=self.device)
            else:
                frames = inpaint_video_with_builded_propainter(
                    self.inpainter, pil_frames, pil_masks, device=self.device,
                    fp16=self.vi_fp16)
        else:
            raise NotImplementedError
        return [np.array(f) if isinstance(f, Image.Image) else f for f in frames]

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
            key_frame_text: str = None,
            text_confidence: float = 0.5,
            video_path: str = None,
    ):
        """
        Mask is 0-1 ndarray in default
        Frame is 0-255 ndarray in default
        """
        assert key_frame_idx == 0, "Only support key frame at the beginning."

        if self.tracker_target == "sam3":
            all_frame, all_mask, all_box = self._track_with_sam3(
                frame_ps, video_path, key_frame_point_coords,
                key_frame_point_labels, key_frame_mask_idx, dilate_kernel_size,
                key_frame_text, text_confidence)
        else:
            all_frame, all_mask, all_box = self._track_with_ostrack(
                frame_ps, key_frame_idx, key_frame_point_coords,
                key_frame_point_labels, key_frame_mask_idx, dilate_kernel_size,
                key_frame_text, text_confidence)

        print("Inpainting ...")
        all_frame = self.forward_inpainter(all_frame, all_mask)
        return all_frame, all_mask, all_box

    def _track_with_sam3(self, frame_ps, video_path, point_coords, point_labels,
                         mask_idx, dilate_kernel_size, text, text_confidence):
        """Let SAM 3 propagate the first-frame selection across the video."""
        if video_path is None:
            raise ValueError(
                "The sam3 tracker needs the source video path; pass "
                "video_path=... or use --tracker ostrack.")

        seed = {}
        if text:
            # Tracking follows one object, so keep the best-scoring instance
            # rather than merging everything the phrase matched.
            key_masks, _ = self.forward_segmentor_text(
                iio.imread(frame_ps[0]), text, text_confidence)
            seed["init_mask"] = key_masks[0]
        elif mask_idx is not None:
            # Reproduce the old granularity choice by seeding with one of SAM's
            # three first-frame candidates.
            key_masks, key_scores = self.forward_segmentor(
                iio.imread(frame_ps[0]), point_coords, point_labels)
            seed["init_mask"] = key_masks[mask_idx]
        else:
            seed["point_coords"] = point_coords
            seed["point_labels"] = point_labels

        print("Tracking with SAM 3 ...")
        raw_masks = track_masks_in_video(self.tracker, video_path, **seed)

        all_frame = [iio.imread(p) for p in frame_ps[:len(raw_masks)]]
        all_mask, all_box = [], []
        for m in raw_masks:
            m = m.astype(np.uint8)
            if dilate_kernel_size is not None:
                m = dilate_mask(m, dilate_kernel_size)
            all_mask.append(m)
            all_box.append(self.get_box_from_mask(m))
        return all_frame, all_mask, all_box

    def _track_with_ostrack(self, frame_ps, key_frame_idx, point_coords,
                            point_labels, mask_idx, dilate_kernel_size,
                            text, text_confidence):
        """Original path: OSTrack follows a box, SAM re-segments every frame."""
        key_frame = iio.imread(frame_ps[key_frame_idx])
        if text:
            key_masks, _ = self.forward_segmentor_text(
                key_frame, text, text_confidence)
            key_mask = key_masks[0]
        else:
            key_masks, key_scores = self.forward_segmentor(
                key_frame, point_coords, point_labels)
            if mask_idx is not None:
                key_mask = key_masks[mask_idx]
            else:
                key_mask = self.mask_selection(key_masks, key_scores)

        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)

        key_box = self.get_box_from_mask(key_mask)

        print("Tracking with OSTrack ...")
        all_box = self.forward_tracker(frame_ps, key_box)

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
    """Example usage (SAM 3 tracking + ProPainter, no extra checkpoints):
    python remove_anything_video.py \
        --input_video ./example/video/paragliding/original_video.mp4 \
        --coords_type key_in \
        --point_coords 652 162 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_ckpt ./pretrained_models/sam3.pt

    Or pick the target in the first frame with a text phrase:
    python remove_anything_video.py \
        --input_video ./example/video/paragliding/original_video.mp4 \
        --text_select "paraglider" \
        --dilate_kernel_size 15 \
        --output_dir ./results

    Legacy path (OSTrack tracking + STTN inpainting):
    python remove_anything_video.py \
        --input_video ./example/video/paragliding/original_video.mp4 \
        --coords_type key_in --point_coords 652 162 --point_labels 1 \
        --output_dir ./results \
        --tracker ostrack --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
        --vi_model sttn --vi_ckpt ./pretrained_models/sttn.pth \
        --mask_idx 2
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
    
    point_coords, point_labels = None, None
    if not args.text_select:
        if args.point_coords is None:
            raise ValueError(
                "Provide either --point_coords or --text_select to pick a target."
            )
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
            dilate_kernel_size, args.text_select, args.text_confidence,
            video_path=video_raw_p
        )
    # visual removed results
    iio.mimwrite(video_rm_w_mask_p, all_frame_rm_w_mask, fps=fps)

    # visual mask
    all_frame = all_frame[:len(all_mask)]
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
