"""Per-frame mask tracking with SAM 3's video predictor.

Replaces the original OSTrack + per-frame SAM combination: SAM 3 propagates the
first-frame mask through the video directly, so there is no separate tracker
checkpoint and no box round-trip that loses mask detail.
"""
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sam3_utils import resolve_sam3_ckpt, warn_if_multiplex_ckpt, DEFAULT_SAM3_CKPT

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def build_sam3_video_tracker(ckpt_p: Optional[str] = None, device="cuda"):
    """Build SAM 3's video tracker.

    The tracker needs the detector's backbone wired in, matching the official
    sam3_for_sam2_video_task example.
    """
    from sam3.model_builder import build_sam3_video_model

    warn_if_multiplex_ckpt(ckpt_p)
    local_ckpt = resolve_sam3_ckpt(ckpt_p)
    try:
        model = build_sam3_video_model(
            checkpoint_path=local_ckpt,
            load_from_HF=local_ckpt is None,
            device=device,
        )
    except Exception as e:
        if local_ckpt is None:
            raise RuntimeError(
                f"Failed to obtain SAM 3 weights automatically ({e}).\n"
                f"Download 'sam3.pt' from https://huggingface.co/facebook/sam3 "
                f"and place it at '{DEFAULT_SAM3_CKPT}', or pass --sam_ckpt."
            ) from e
        raise

    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    # The video model already contains the detector, which is all a text prompt
    # needs. Exposing it here saves callers from loading a second 3.4GB copy of
    # SAM 3 just to name the target — that duplication is enough to OOM a 24GB
    # card once ProPainter's activations are added on top.
    from sam_segment import Sam3PredictorAdapter
    predictor.text_segmentor = Sam3PredictorAdapter(model.detector, device=device)
    return predictor


@contextmanager
def _as_frame_folder(video_path: str):
    """Yield a folder of `<index>.jpg` frames for SAM 3 to read.

    SAM 3 decodes mp4 through `decord`, which is an awkward dependency to
    guarantee. Going through a JPEG folder avoids it, and lets frame-sequence
    datasets (Aria, TACO, Ego4D exports) be passed in directly.
    """
    src = Path(video_path)
    if src.is_dir():
        names = sorted(p for p in src.iterdir() if p.suffix in IMAGE_EXTS)
        if not names:
            raise ValueError(f"No image frames found in {src}")
        # SAM 3 sorts by int(stem), so re-link to that naming when needed.
        try:
            sorted(names, key=lambda p: int(p.stem))
            if all(p.suffix.lower() in (".jpg", ".jpeg") for p in names):
                yield str(src)
                return
        except ValueError:
            pass
        tmp = Path(tempfile.mkdtemp(prefix="ia_frames_"))
        try:
            import imageio.v2 as iio
            for i, p in enumerate(names):
                iio.imwrite(tmp / f"{i:05d}.jpg", iio.imread(p), quality=95)
            yield str(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    import imageio.v2 as iio
    tmp = Path(tempfile.mkdtemp(prefix="ia_frames_"))
    try:
        reader = iio.get_reader(str(src))
        for i, frame in enumerate(reader):
            iio.imwrite(tmp / f"{i:05d}.jpg", frame, quality=95)
        reader.close()
        yield str(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@torch.no_grad()
def track_masks_in_video(
        predictor,
        video_path: str,
        init_mask: Optional[np.ndarray] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        max_frames: Optional[int] = None,
        offload_to_cpu: bool = False,
) -> List[np.ndarray]:
    """Propagate a first-frame prompt across the whole video.

    `video_path` is an mp4 file or a folder of JPEG frames. Exactly one of
    `init_mask`, `point_coords` or `box` seeds the track.

    Returns one bool mask of shape (H, W) per frame, in frame order.
    """
    with _as_frame_folder(video_path) as frame_dir:
        return _track(predictor, frame_dir, init_mask, point_coords,
                      point_labels, box, frame_idx, max_frames, offload_to_cpu)


def _track(predictor, frame_dir, init_mask, point_coords, point_labels, box,
           frame_idx, max_frames, offload_to_cpu=False):
    state = predictor.init_state(
        video_path=frame_dir,
        offload_video_to_cpu=offload_to_cpu,
        offload_state_to_cpu=offload_to_cpu,
    )
    obj_id = 1

    if init_mask is not None:
        mask_t = torch.as_tensor(np.asarray(init_mask) > 0, dtype=torch.bool)
        predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id,
                               mask=mask_t)
    elif point_coords is not None or box is not None:
        h, w = state["video_height"], state["video_width"]
        points_t = labels_t = box_t = None
        if point_coords is not None:
            # The video predictor expects coordinates normalized to [0, 1],
            # unlike the image predictor which takes pixels.
            rel = [[x / w, y / h] for x, y in np.asarray(point_coords, dtype=float)]
            points_t = torch.tensor(rel, dtype=torch.float32)
            labels_t = torch.tensor(np.asarray(point_labels), dtype=torch.int32)
        if box is not None:
            x0, y0, x1, y1 = np.asarray(box, dtype=float)
            box_t = torch.tensor([x0 / w, y0 / h, x1 / w, y1 / h],
                                 dtype=torch.float32)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=frame_idx, obj_id=obj_id,
            points=points_t, labels=labels_t, box=box_t,
            clear_old_points=False,
        )
    else:
        raise ValueError(
            "Provide one of init_mask, point_coords or box to seed tracking.")

    num_frames = state["num_frames"]
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    per_frame = {}
    for fidx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=num_frames,
            reverse=False, propagate_preflight=True):
        idx = obj_ids.index(obj_id) if obj_id in obj_ids else 0
        per_frame[fidx] = (video_res_masks[idx] > 0.0).squeeze().cpu().numpy()

    return [per_frame[i] for i in sorted(per_frame)]
