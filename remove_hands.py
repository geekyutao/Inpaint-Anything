"""Batch hand/human removal for egocentric video, for Human-to-Robot data pipelines.

This is the "hand removal and inpainting" stage of a Human-to-Robot synthesis
pipeline (see Qwen-RobotManip) and the erasure half of EgoEngine-style visual
generation. It does *not* do action retargeting, robot rendering or depth
compositing — it produces clean plates plus the exact masks those later stages
need to composite into.

It walks a dataset directory, finds every episode (an mp4 file or a folder of
frames), erases the target with SAM 3 + ProPainter, and mirrors the input layout
in the output directory. Models are loaded once and reused across episodes, and
finished episodes are skipped so an interrupted run can be resumed.

Example:
    python remove_hands.py \
        --input_dir /data/ego4d/clips \
        --output_dir /data/ego4d_handless \
        --save_masks
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from propainter_video_inpaint import (
    DEFAULT_PROPAINTER_CKPT_DIR,
    build_propainter_model,
    inpaint_video_with_builded_propainter,
)
from sam3_utils import DEFAULT_SAM3_CKPT
from sam3_video_track import (
    IMAGE_EXTS,
    _as_frame_folder,
    build_sam3_video_tracker,
    track_masks_in_video,
)
from utils import dilate_mask

VIDEO_EXTS = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".webm")

# Erasing the forearm as well is the usual choice for Human-to-Robot synthesis,
# because the rendered robot arm covers that region anyway.
DEFAULT_TARGET = "hands and forearms"
TARGET_WITH_OBJECT = "hands, forearms and the object being held"


@dataclass
class Episode:
    name: str            # path relative to the dataset root, used for mirroring
    source: Path
    is_video: bool
    frame_names: List[str] = field(default_factory=list)


# Directories that hold this tool's own output, not source episodes. Without
# this, re-running over a tree that already contains results would re-process
# them.
DEFAULT_EXCLUDES = ("_masks", "_plates", "comparison")


def discover_episodes(root: Path, excludes=DEFAULT_EXCLUDES) -> List[Episode]:
    """Find every episode under `root`.

    Handles the two layouts common to egocentric datasets: one video file per
    episode (Ego4D-style) and one folder of frames per episode (EPIC-KITCHENS,
    TACO, Aria exports).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    # A single episode passed directly.
    if root.is_file():
        return [Episode(root.stem, root, True)]
    if _frames_in(root):
        return [Episode(root.name, root, False, _frames_in(root))]

    def excluded(p: Path) -> bool:
        parts = p.relative_to(root).parts
        return any(part in excludes or part.startswith("_") for part in parts)

    episodes = []
    for p in sorted(root.rglob("*")):
        if excluded(p):
            continue
        if p.is_file() and p.suffix in VIDEO_EXTS:
            episodes.append(Episode(str(p.relative_to(root).with_suffix("")), p, True))
        elif p.is_dir():
            frames = _frames_in(p)
            if frames:
                episodes.append(Episode(str(p.relative_to(root)), p, False, frames))
    return episodes


def _frames_in(d: Path) -> List[str]:
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_file() and p.suffix in IMAGE_EXTS)


def _read_frames(ep: Episode) -> List[Image.Image]:
    """Load an episode's frames from the original source.

    Deliberately not read from the normalized folder handed to SAM 3: that one
    is re-encoded JPEG, and feeding it to the inpainter would bake an extra
    generation of compression loss into the clean plates. The tracker only
    needs it to produce masks.
    """
    if ep.is_video:
        import imageio.v2 as iio
        return [Image.fromarray(f).convert("RGB")
                for f in iio.mimread(str(ep.source), memtest=False)]
    return [Image.open(ep.source / n).convert("RGB") for n in ep.frame_names]


def _smooth_masks(masks: List[np.ndarray], radius: int) -> List[np.ndarray]:
    """Fill short gaps where tracking dropped the target.

    Hands leave and re-enter the frame constantly in egocentric footage, and a
    single dropped frame leaves a visible flash of the original hand. A frame
    with no detection surrounded by detections is filled from its neighbours.
    """
    if radius <= 0:
        return masks
    areas = np.array([m.sum() for m in masks])
    out = list(masks)
    for i, a in enumerate(areas):
        if a > 0:
            continue
        lo = max(0, i - radius)
        hi = min(len(masks), i + radius + 1)
        neighbours = [masks[j] for j in range(lo, hi) if areas[j] > 0]
        if neighbours:
            out[i] = np.logical_or.reduce(neighbours)
    return out


def _seed_masks(tracker, frame_dir, target, confidence, segmentor,
                offload) -> Optional[List[np.ndarray]]:
    """Segment `target` on the first frame that contains it, then propagate.

    Egocentric clips often start with the hands out of view, so a failure on
    frame 0 is not a failure of the episode.
    """
    names = sorted(p for p in Path(frame_dir).iterdir() if p.suffix in IMAGE_EXTS)
    probe_idxs = [0]
    probe_idxs += [int(len(names) * f) for f in (0.1, 0.25, 0.5)]
    seen = set()

    for idx in probe_idxs:
        if idx in seen or idx >= len(names):
            continue
        seen.add(idx)
        frame = np.array(Image.open(names[idx]).convert("RGB"))
        segmentor.set_image(frame)
        try:
            masks, _, _ = segmentor.predict_text(
                target, confidence_threshold=confidence)
        finally:
            segmentor.reset_image()
        if len(masks) == 0:
            continue
        init_mask = np.any(masks, axis=0)
        return track_masks_in_video(
            tracker, str(frame_dir), init_mask=init_mask, frame_idx=idx,
            offload_to_cpu=offload)
    return None


def process_episode(models, ep: Episode, args) -> dict:
    """Erase the target from one episode. Returns a manifest entry."""
    started = time.time()
    out_dir = Path(args.output_dir)
    dst = out_dir / (ep.name + (".mp4" if ep.is_video else ""))
    mask_dst = Path(args.mask_dir or (out_dir / "_masks")) / ep.name

    entry = {"name": ep.name, "source": str(ep.source), "output": str(dst)}

    if args.skip_existing and dst.exists():
        entry.update(status="skipped")
        return entry

    with _as_frame_folder(str(ep.source)) as frame_dir:
        frames = _read_frames(ep)
        entry["frames"] = len(frames)
        if not frames:
            entry.update(status="empty")
            return entry

        raw_masks = _seed_masks(
            models["tracker"], frame_dir, args.target, args.text_confidence,
            models["segmentor"], args.offload)

        if raw_masks is None:
            entry.update(status="no_target",
                         note=f"{args.target!r} not found in any probed frame")
            return entry

        raw_masks = [m.astype(bool) for m in raw_masks[:len(frames)]]
        raw_masks = _smooth_masks(raw_masks, args.mask_gap_fill)

        pil_masks = []
        for m in raw_masks:
            m8 = m.astype(np.uint8)
            if args.dilate_kernel_size:
                m8 = dilate_mask(m8, args.dilate_kernel_size)
            pil_masks.append(Image.fromarray(np.uint8(m8 * 255)))

        coverage = float(np.mean([np.mean(np.array(m) > 0) for m in pil_masks]))
        entry["mask_coverage"] = round(coverage, 5)
        if coverage < args.min_coverage:
            entry.update(status="low_coverage")
            if not args.keep_low_coverage:
                return entry

        frames = frames[:len(pil_masks)]
        out_frames = _inpaint_chunked(models["painter"], frames, pil_masks, args)

        _write_output(out_frames, ep, dst, args)
        if args.save_masks:
            _write_masks(pil_masks, mask_dst, args.mask_format)
            entry["masks"] = str(mask_dst)

    entry.update(status="ok", seconds=round(time.time() - started, 1))
    return entry


def _inpaint_chunked(painter, frames, masks, args):
    """Run ProPainter, optionally in overlapping chunks to bound memory."""
    n = len(frames)
    if args.chunk_size <= 0 or n <= args.chunk_size:
        return inpaint_video_with_builded_propainter(
            painter, frames, masks, device=args.device, fp16=args.fp16,
            subvideo_length=args.subvideo_length)

    overlap = args.chunk_overlap
    out = [None] * n
    start = 0
    while start < n:
        end = min(n, start + args.chunk_size)
        lo = max(0, start - overlap)
        chunk = inpaint_video_with_builded_propainter(
            painter, frames[lo:end], masks[lo:end], device=args.device,
            fp16=args.fp16, subvideo_length=args.subvideo_length)
        # Discard the warm-up overlap, keep the freshly computed tail.
        for i, f in enumerate(chunk):
            idx = lo + i
            if idx >= start:
                out[idx] = f
        start = end
    return out


def _write_output(out_frames, ep: Episode, dst: Path, args):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if ep.is_video:
        import imageio.v2 as iio
        import imageio
        try:
            fps = imageio.v3.immeta(str(ep.source),
                                    exclude_applied=False).get("fps", args.fps)
        except Exception:
            fps = args.fps
        iio.mimwrite(dst, [np.array(f) for f in out_frames], fps=fps,
                     quality=args.video_quality)
    else:
        dst.mkdir(parents=True, exist_ok=True)
        # Preserve the original frame filenames so downstream stages can join
        # on them (poses, annotations, retargeted actions).
        names = ep.frame_names[:len(out_frames)]
        for name, frame in zip(names, out_frames):
            out_name = name
            if args.output_format == "png":
                out_name = str(Path(name).with_suffix(".png"))
            elif args.output_format == "jpg":
                out_name = str(Path(name).with_suffix(".jpg"))
            p = dst / out_name
            if p.suffix.lower() in (".jpg", ".jpeg"):
                frame.save(p, quality=args.jpeg_quality, subsampling=0)
            else:
                frame.save(p)


def _write_masks(pil_masks, mask_dst: Path, fmt: str):
    mask_dst.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        arr = np.stack([np.array(m) > 0 for m in pil_masks])
        np.savez_compressed(mask_dst.with_suffix(".npz"), masks=arr)
    else:
        for i, m in enumerate(pil_masks):
            m.save(mask_dst / f"{i:06d}.png")


def build_models(args):
    device = args.device
    tracker = build_sam3_video_tracker(args.sam_ckpt, device=device)
    # Reuse the detector that ships inside the video model rather than loading
    # SAM 3 twice; see build_sam3_video_tracker.
    segmentor = tracker.text_segmentor
    painter = build_propainter_model(args.propainter_ckpt, device=device)
    return {"tracker": tracker, "segmentor": segmentor, "painter": painter}


def setup_args(parser):
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Dataset root, a single episode folder, or a single video file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write clean plates, mirroring the input layout.",
    )
    parser.add_argument(
        "--mode", type=str, default="keep_object",
        choices=["keep_object", "remove_object"],
        help="'keep_object' erases only the hands and forearms, leaving the "
             "manipulated object in place. 'remove_object' erases the held "
             "object too, for pipelines that re-render it. Default: keep_object",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help=f"Override what gets erased. Defaults to {DEFAULT_TARGET!r} "
             f"({TARGET_WITH_OBJECT!r} in remove_object mode).",
    )
    parser.add_argument(
        "--text_confidence", type=float, default=0.4,
        help="SAM 3 detection threshold. Lower than the default elsewhere "
             "because hands are often partly out of frame. Default: 0.4",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=12,
        help="Grow the mask to swallow contact shadows and soft edges.",
    )
    parser.add_argument(
        "--mask_gap_fill", type=int, default=2,
        help="Fill single-frame tracking dropouts from neighbours within this "
             "radius. 0 disables.",
    )
    parser.add_argument(
        "--min_coverage", type=float, default=0.0005,
        help="Flag episodes whose average mask covers less than this fraction "
             "of the frame; usually means segmentation failed.",
    )
    parser.add_argument(
        "--keep_low_coverage", action="store_true",
        help="Process low-coverage episodes anyway instead of flagging only.",
    )
    parser.add_argument(
        "--save_masks", action="store_true",
        help="Export the masks; depth-guided compositing downstream needs them.",
    )
    parser.add_argument(
        "--mask_dir", type=str, default=None,
        help="Where masks go. Default: <output_dir>/_masks",
    )
    parser.add_argument(
        "--mask_format", type=str, default="png", choices=["png", "npz"],
        help="Lossless per-frame PNGs, or one compressed npz per episode.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=0,
        help="Inpaint in chunks of this many frames to bound GPU memory. "
             "0 (default) processes the episode in one pass, which preserves "
             "long-range consistency.",
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=10,
        help="Warm-up frames shared between chunks.",
    )
    parser.add_argument(
        "--subvideo_length", type=int, default=80,
        help="ProPainter's internal sub-video length.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Half precision for ProPainter.",
    )
    parser.add_argument(
        "--offload", action="store_true",
        help="Keep SAM 3's frame tensors on CPU. Slower, but needed for long "
             "episodes.",
    )
    parser.add_argument(
        "--output_format", type=str, default="same",
        choices=["same", "png", "jpg"],
        help="Frame output format. 'png' is lossless — worth it when the clean "
             "plates feed further compositing, since re-encoding to JPEG "
             "accumulates loss on every pass. Default: same as input.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="Quality for JPEG frame output.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Fallback FPS when the source has no metadata.")
    parser.add_argument("--video_quality", type=int, default=8,
                        help="imageio quality for mp4 output (0-10).")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Resume: skip episodes whose output already exists.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N episodes. 0 = all.")
    parser.add_argument("--exclude", nargs="+", default=list(DEFAULT_EXCLUDES),
                        help="Directory names to skip during discovery. Names "
                             "starting with '_' are always skipped.")
    parser.add_argument(
        "--sam_ckpt", type=str, default=DEFAULT_SAM3_CKPT,
        help="SAM 3 checkpoint.",
    )
    parser.add_argument(
        "--propainter_ckpt", type=str, default=DEFAULT_PROPAINTER_CKPT_DIR,
        help="Directory holding the ProPainter checkpoints.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="List the episodes that would be processed and exit.")
    parser.add_argument("--traceback", action="store_true",
                        help="Print full tracebacks for failed episodes.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch hand/human removal for egocentric video.")
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.target is None:
        args.target = (TARGET_WITH_OBJECT if args.mode == "remove_object"
                       else DEFAULT_TARGET)

    episodes = discover_episodes(Path(args.input_dir), tuple(args.exclude))
    if args.limit:
        episodes = episodes[:args.limit]
    print(f"Found {len(episodes)} episode(s) under {args.input_dir}")
    print(f"Erasing: {args.target!r}")

    if args.dry_run:
        for ep in episodes:
            kind = "video" if ep.is_video else f"{len(ep.frame_names)} frames"
            print(f"  {ep.name}  ({kind})")
        return

    if not episodes:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(args)

    manifest = []
    counts = {}
    for i, ep in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] {ep.name}", flush=True)
        try:
            entry = process_episode(models, ep, args)
        except Exception as e:
            entry = {"name": ep.name, "source": str(ep.source),
                     "status": "error", "error": f"{type(e).__name__}: {e}"}
            print(f"    failed: {entry['error']}", flush=True)
            if args.traceback:
                import traceback
                traceback.print_exc()
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        manifest.append(entry)
        (out_dir / "manifest.json").write_text(
            json.dumps({"target": args.target, "episodes": manifest}, indent=2))

    print("\nDone. " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
