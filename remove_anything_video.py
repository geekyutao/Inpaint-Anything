import torch
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from segment_anything import SamPredictor, sam_model_registry
from sam_segment import build_sam_model
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from tracking_lib.test.evaluation.video2seq import video2seq
from utils import load_img_to_array, save_array_to_img


class RemoveAnythingVideo(nn.Module):
    def __init__(
            self,
            segmentor_target="sam",
            segmentor_build_args: Dict = None,
            inpainter_target="lama",
            inpainter_build_args: Dict = None,
    ):
        super().__init__()
        if segmentor_build_args is None:
            segmentor_build_args = {
                "model_type": "vit_h",
                "ckpt_p": "./pretrained_models/sam_vit_h_4b8939.pth"
            }
        if inpainter_build_args is None:
            inpainter_build_args = {
                "config_p": "./lama/configs/prediction/default.yaml",
                "ckpt_p": "./pretrained_models/big-lama"
            }
        # self.tracker = self.build_tracker()
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args)


    def build_tracker(self, target, **kwargs):
        raise NotImplementedError

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="lama", **kwargs):
        assert target == "lama", "Only support lama now."
        return build_lama_model(**kwargs, device=self.device)

    def forward_tracker(self, **kwargs):
        raise NotImplementedError

    def forward_segmentor(self, img, point_coords=None, point_labels=None,
                          box=None, mask_input=None, multimask_output=True,
                          return_logits=False):
        self.segmentor.set_image(img)
        masks, scores, logits = self.segmentor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, img, mask):
        return inpaint_img_with_builded_lama(
            self.inpainter, img, mask, device=self.device)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def mask_selection(masks, scores, ref_mask=None, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            if ref_mask is not None:
                idx = (masks - ref_mask).abs().sum(-1).sum(-1).argmin()
            else:
                idx = scores.argmax()
            return masks[idx]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return [x, y, x + w, y + h]

    def forward(
            self,
            all_frame,
            key_frame_idx,
            key_frame_point_coords,
            key_frame_point_labels,
    ):
        # get key-frame mask
        assert key_frame_idx == 0, "Only support key frame at the beginning."
        key_frame = all_frame[key_frame_idx]
        key_masks, key_scores = self.forward_segmentor(
            key_frame, key_frame_point_coords, key_frame_point_labels)
        key_mask = self.mask_selection(key_masks, key_scores)
        key_mask = (key_mask * 255).astype(np.uint8)

        # get key-frame box
        key_box = self.get_box_from_mask(key_mask)

        # get all-frame boxes using video tracker
        all_box = self.forward_tracker(all_frame, key_box)

        # get all-frame masks using sam
        all_mask = []
        ref_mask = key_mask
        for frame, box in zip(all_frame, all_box):
            masks, scores = self.forward_segmentor(frame, box=box)
            mask = self.mask_selection(masks, scores, ref_mask)
            ref_mask = mask
            all_mask.append(mask)

        # get all-frame inpainted results
        all_frame = self.inpainter(all_frame, all_mask)
        return all_frame, all_mask


if __name__ == "__main__":
    frame_ps = sorted(glob.glob(
        "/data1/yutao/projects/Inpaint-Anything/results/original_frames/*"))
    frames = [load_img_to_array(p) for p in frame_ps]
    model = RemoveAnythingVideo()
    model(frames, 0, [[300, 300]], [1])



