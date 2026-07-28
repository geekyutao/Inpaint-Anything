import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
import torch
from PIL import Image

from sam3_utils import DEFAULT_SAM3_CKPT, build_sam3_image_model, \
    sam3_autocast as _autocast
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points


SAM3_MODEL_TYPES = ("sam3",)
SAM1_MODEL_TYPES = ("vit_h", "vit_l", "vit_b", "vit_t")
ALL_MODEL_TYPES = SAM3_MODEL_TYPES + SAM1_MODEL_TYPES

_build_sam3_image_model = build_sam3_image_model


class Sam3PredictorAdapter:
    """Expose SAM 3 through the SAM 1 `SamPredictor` API used across this repo.

    SAM 3 splits what SAM 1 did in one object: `Sam3Processor.set_image` runs the
    shared backbone and returns an inference state, and `model.predict_inst`
    consumes that state with SAM 1's exact `predict()` signature. The state also
    doubles as the cached-embedding handle that `app.py` stores between clicks,
    so it is what `features` maps onto.
    """

    def __init__(self, model, device="cuda", confidence_threshold=0.5):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = model
        self._device = device
        self._interactive = getattr(model, "inst_interactive_predictor", None) is not None
        self._processor = Sam3Processor(model, device=device,
                                        confidence_threshold=confidence_threshold)
        self._state = None
        # SAM 3 resizes every input to a fixed square; SAM 1 exposed a padded
        # input size that app.py caches next to the features.
        self._input_size = self._processor.resolution

    def set_image(self, image) -> None:
        # Sam3Processor derives H/W via shape[-2:], which is wrong for HWC
        # arrays, so always hand it a PIL image.
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        with _autocast(self._device):
            self._state = self._processor.set_image(pil_img)

    def predict(self, point_coords=None, point_labels=None, box=None,
                mask_input=None, multimask_output=True, return_logits=False):
        if self._state is None:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        if not self._interactive:
            raise RuntimeError(
                "This SAM 3 model was built without instance interactivity; "
                "rebuild it with enable_inst_interactivity=True for point and "
                "box prompts."
            )
        with _autocast(self._device):
            return self.model.predict_inst(
                self._state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )

    def predict_text(self, text_prompt: str, confidence_threshold=None):
        """Segment all instances matching an open-vocabulary phrase.

        Returns (masks (N,H,W) bool, scores (N,), boxes (N,4) XYXY), sorted by
        descending score.
        """
        if self._state is None:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        if confidence_threshold is not None:
            self._processor.confidence_threshold = confidence_threshold
        with _autocast(self._device):
            self._state = self._processor.set_text_prompt(text_prompt, self._state)

        masks = self._state["masks"]
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.detach().cpu().numpy().astype(bool)
        scores = self._state["scores"].float().detach().cpu().numpy()
        boxes = self._state["boxes"].float().detach().cpu().numpy()
        order = np.argsort(-scores)
        return masks[order], scores[order], boxes[order]

    def reset_image(self) -> None:
        self._state = None

    def to(self, device):
        self._device = device
        return self

    @property
    def device(self):
        return self._device

    @property
    def features(self):
        return self._state

    @features.setter
    def features(self, value):
        self._state = value

    @property
    def is_image_set(self):
        return self._state is not None

    @is_image_set.setter
    def is_image_set(self, value):
        if not value:
            self._state = None

    @property
    def orig_h(self):
        return self._state["original_height"] if self._state else None

    @orig_h.setter
    def orig_h(self, value):
        if self._state is not None:
            self._state["original_height"] = value

    @property
    def orig_w(self):
        return self._state["original_width"] if self._state else None

    @orig_w.setter
    def orig_w(self, value):
        if self._state is not None:
            self._state["original_width"] = value

    @property
    def input_h(self):
        return self._input_size

    @input_h.setter
    def input_h(self, value):
        self._input_size = value

    @property
    def input_w(self):
        return self._input_size

    @input_w.setter
    def input_w(self, value):
        self._input_size = value


def build_sam_model(model_type: str, ckpt_p: str, device="cuda"):
    """Build a predictor exposing the SAM 1 `SamPredictor` interface.

    `model_type` selects the backend: "sam3" or a SAM 1 / MobileSAM variant
    ("vit_h", "vit_l", "vit_b", "vit_t").
    """
    if model_type in SAM3_MODEL_TYPES:
        model = _build_sam3_image_model(ckpt_p, device,
                                        enable_inst_interactivity=True)
        return Sam3PredictorAdapter(model, device=device)

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        model_type: str,
        ckpt_p: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    predictor = build_sam_model(model_type, ckpt_p, device=device)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def predict_masks_with_text(
        img: np.ndarray,
        text_prompt: str,
        ckpt_p: str = DEFAULT_SAM3_CKPT,
        device="cuda",
        confidence_threshold: float = 0.5,
        predictor=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment every instance matching an open-vocabulary phrase with SAM 3.

    Unlike the point-prompted path, which returns three candidate granularities
    for one object, this returns one mask per detected instance.

    Returns (masks, scores, boxes) with masks as a bool array of shape (N, H, W)
    and boxes in XYXY pixel coordinates, sorted by descending score.
    """
    if predictor is None:
        model = _build_sam3_image_model(ckpt_p, device,
                                        enable_inst_interactivity=False)
        predictor = Sam3PredictorAdapter(
            model, device=device, confidence_threshold=confidence_threshold)

    predictor.set_image(img)
    masks, scores, boxes = predictor.predict_text(
        text_prompt, confidence_threshold=confidence_threshold)

    if len(scores) == 0:
        raise ValueError(
            f"SAM 3 found no object matching the text prompt {text_prompt!r}. "
            f"Try a simpler noun phrase or lower --text_confidence."
        )
    return masks, scores, boxes


def select_masks(
        img: np.ndarray,
        model_type: str,
        ckpt_p: str,
        device="cuda",
        point_coords: Optional[List[float]] = None,
        point_labels: Optional[List[int]] = None,
        text_select: Optional[str] = None,
        text_confidence: float = 0.5,
) -> np.ndarray:
    """Segment the target either by clicked point or by an SAM 3 text phrase.

    Point prompts yield SAM's three candidate granularities; a text phrase
    instead yields every matching instance, which is merged into one mask so
    that "remove the cars" removes all of them in a single pass.

    Returns a uint8 array of shape (N, H, W) with values in {0, 255}.
    """
    if text_select:
        if model_type not in SAM3_MODEL_TYPES:
            raise ValueError(
                f"--text_select requires --sam_model_type sam3, got {model_type!r}."
            )
        masks, _, _ = predict_masks_with_text(
            img, text_select, ckpt_p=ckpt_p, device=device,
            confidence_threshold=text_confidence,
        )
        merged = np.any(masks, axis=0)[None, ...]
        return merged.astype(np.uint8) * 255

    if point_coords is None:
        raise ValueError(
            "Provide either --point_coords or --text_select to pick a target."
        )
    masks, _, _ = predict_masks_with_sam(
        img, [point_coords], point_labels,
        model_type=model_type, ckpt_p=ckpt_p, device=device,
    )
    return masks.astype(np.uint8) * 255


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
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
        default="sam3", choices=list(ALL_MODEL_TYPES),
        help="The type of sam model to load. Default: 'sam3'",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, default=DEFAULT_SAM3_CKPT,
        help="The path to the SAM checkpoint to use for mask generation.",
    )


if __name__ == "__main__":
    """Example usage:
    python sam_segment.py \
        --input_img FA_demo/FA1_dog.png \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "sam3" \
        --sam_ckpt ./pretrained_models/sam3.pt
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)

    masks, _, _ = predict_masks_with_sam(
        img,
        [args.point_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

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
        show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()
