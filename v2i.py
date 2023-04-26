import torch
import numpy as np
import cv2
import torch.nn as nn
from typing import Any, Dict, List
from segment_anything import SamPredictor, sam_model_registry
from sam_segment import build_sam_model
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from tracking_lib.test.evaluation.video2seq import video2seq

# video_seq, fps = video2seq(
#     '/data1/yutao/projects/Inpaint-Anything/example/remove-anything-video/ikun.mp4',
#     [290, 341],
#     [1],
#     "vit_h",
#     '/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth',
#     './results'
# )
#

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
        assert target == "lama", "Only support sam now."
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
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        self.segmentor.reset_image()
        return masks

    def forward_inpainter(self, img, mask):
        return inpaint_img_with_builded_lama(
            self.inpainter, img, mask, device=self.device)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def mask_selection(self, masks, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            return masks[0]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return [x, y, x + w, y + h]

    def forward(
            self,
            frames,
            key_frame_idx,
            key_frame_point_coords,
            key_frame_point_labels
    ):
        key_frame = frames[key_frame_idx]
        masks = self.forward_segmentor(
            key_frame, key_frame_point_coords, key_frame_point_labels)
        mask = self.mask_selection(masks, interactive=False)
        # return self.forward_inpainter(img, masks)
        box = self.get_box_from_mask(mask)




print(video_seq.ground_truth_rect, fps)
