"""ProPainter (ICCV 2023) video inpainting, wrapped to match this repo's API.

Mirrors the interface of `sttn_video_inpaint.py` so the two backends are
interchangeable from `remove_anything_video.py`.

Note: ProPainter and STTN both ship top-level `model` / `core` packages, so
their imports are deferred until a backend is actually selected. Only one video
inpainting backend can be used per process.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from tqdm import tqdm

PROPAINTER_DIR = Path(__file__).resolve().parent / "propainter"
DEFAULT_PROPAINTER_CKPT_DIR = "./pretrained_models/propainter"
PRETRAIN_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
_CKPT_FILES = ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")


def _import_propainter():
    if str(PROPAINTER_DIR) not in sys.path:
        sys.path.insert(0, str(PROPAINTER_DIR))
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    return RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator


def _ensure_ckpt(ckpt_dir: str, filename: str) -> str:
    """Return a local checkpoint path, downloading it on first use."""
    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / filename
    if not dst.exists():
        from torch.hub import download_url_to_file
        print(f"Downloading {filename} -> {dst}")
        download_url_to_file(PRETRAIN_URL + filename, str(dst), progress=True)
    return str(dst)


def _to_tensor(pil_list: List[Image.Image]) -> torch.Tensor:
    """Stack PIL frames into a (T, C, H, W) float tensor in [0, 1].

    Reimplements ProPainter's `core.utils.to_tensors` so that importing its
    `core` package (which clashes with STTN's) can be avoided.
    """
    mode = pil_list[0].mode
    if mode == "1":
        pil_list = [im.convert("L") for im in pil_list]
        mode = "L"
    if mode == "L":
        stacked = np.stack([np.expand_dims(np.array(im), 2) for im in pil_list], axis=2)
    else:
        stacked = np.stack([np.array(im) for im in pil_list], axis=2)
    return torch.from_numpy(stacked).permute(2, 3, 0, 1).contiguous().float().div(255)


def _prepare_masks(masks: List[Image.Image], size, flow_mask_dilates=8, mask_dilates=5):
    """Build the two mask streams ProPainter expects: one for optical flow, one
    for the inpainting region. Mirrors `read_mask` in inference_propainter.py."""
    flow_masks, masks_dilated = [], []
    for mask in masks:
        if size is not None:
            mask = mask.resize(size, Image.NEAREST)
        arr = np.array(mask.convert("L"))

        if flow_mask_dilates > 0:
            flow_arr = scipy.ndimage.binary_dilation(
                arr, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_arr = (arr > 25).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_arr * 255))

        if mask_dilates > 0:
            dil = scipy.ndimage.binary_dilation(
                arr, iterations=mask_dilates).astype(np.uint8)
        else:
            dil = (arr > 25).astype(np.uint8)
        masks_dilated.append(Image.fromarray(dil * 255))
    return flow_masks, masks_dilated


def _get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def build_propainter_model(ckpt_p=None, device="cuda"):
    """Load RAFT, the flow completion net and the ProPainter generator."""
    RAFT_bi, RecurrentFlowCompleteNet, InpaintGenerator = _import_propainter()
    ckpt_dir = ckpt_p or DEFAULT_PROPAINTER_CKPT_DIR
    # A file path is accepted for symmetry with the other backends; ProPainter
    # needs three checkpoints, so only the directory matters.
    if Path(ckpt_dir).suffix:
        ckpt_dir = str(Path(ckpt_dir).parent)

    raft = RAFT_bi(_ensure_ckpt(ckpt_dir, "raft-things.pth"), device)

    flow_complete = RecurrentFlowCompleteNet(
        _ensure_ckpt(ckpt_dir, "recurrent_flow_completion.pth"))
    for p in flow_complete.parameters():
        p.requires_grad = False
    flow_complete.to(device).eval()

    generator = InpaintGenerator(
        model_path=_ensure_ckpt(ckpt_dir, "ProPainter.pth")).to(device)
    generator.eval()

    return {"raft": raft, "flow_complete": flow_complete, "generator": generator}


@torch.no_grad()
def inpaint_video_with_builded_propainter(*args, **kwargs) -> List[Image.Image]:
    from sam3_utils import no_autocast
    with no_autocast():
        return _inpaint_video_with_builded_propainter(*args, **kwargs)


@torch.no_grad()
def _inpaint_video_with_builded_propainter(
        model,
        frames: List[Image.Image],
        masks: List[Image.Image],
        device="cuda",
        resize_ratio: float = 1.0,
        height: int = -1,
        width: int = -1,
        mask_dilation: int = 4,
        ref_stride: int = 10,
        neighbor_length: int = 10,
        subvideo_length: int = 80,
        raft_iter: int = 20,
        fp16: bool = False,
) -> List[Image.Image]:
    """Remove the masked region across a whole video.

    frames/masks are PIL lists at the original resolution; the result comes back
    at that same resolution with untouched pixels preserved exactly.
    """
    raft = model["raft"]
    flow_complete = model["flow_complete"]
    generator = model["generator"]

    use_half = fp16 and str(device).startswith("cuda")

    out_size = frames[0].size
    size = out_size
    if width != -1 and height != -1:
        size = (width, height)
    if resize_ratio != 1.0:
        size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))
    # The network stride requires both sides to be multiples of 8.
    process_size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
    proc_frames = [f.convert("RGB").resize(process_size) for f in frames]
    w, h = process_size

    flow_masks, masks_dilated = _prepare_masks(
        masks, process_size,
        flow_mask_dilates=mask_dilation, mask_dilates=mask_dilation)

    ori_frames = [np.array(f).astype(np.uint8) for f in proc_frames]
    frames_t = _to_tensor(proc_frames).unsqueeze(0) * 2 - 1
    flow_masks_t = _to_tensor(flow_masks).unsqueeze(0)
    masks_dilated_t = _to_tensor(masks_dilated).unsqueeze(0)
    frames_t = frames_t.to(device)
    flow_masks_t = flow_masks_t.to(device)
    masks_dilated_t = masks_dilated_t.to(device)

    video_length = frames_t.size(1)

    # ---- optical flow, chunked to bound memory ----
    if frames_t.size(-1) <= 640:
        short_clip_len = 12
    elif frames_t.size(-1) <= 720:
        short_clip_len = 8
    elif frames_t.size(-1) <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2

    if video_length > short_clip_len:
        fwd, bwd = [], []
        for f in range(0, video_length, short_clip_len):
            end_f = min(video_length, f + short_clip_len)
            start_f = f if f == 0 else f - 1
            flows_f, flows_b = raft(frames_t[:, start_f:end_f], iters=raft_iter)
            fwd.append(flows_f)
            bwd.append(flows_b)
            torch.cuda.empty_cache()
        gt_flows_bi = (torch.cat(fwd, dim=1), torch.cat(bwd, dim=1))
    else:
        gt_flows_bi = raft(frames_t, iters=raft_iter)
        torch.cuda.empty_cache()

    if use_half:
        frames_t = frames_t.half()
        flow_masks_t = flow_masks_t.half()
        masks_dilated_t = masks_dilated_t.half()
        gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
        flow_complete = flow_complete.half()
        generator = generator.half()

    # ---- flow completion ----
    flow_length = gt_flows_bi[0].size(1)
    if flow_length > subvideo_length:
        pred_flows_f, pred_flows_b = [], []
        pad_len = 5
        for f in range(0, flow_length, subvideo_length):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + subvideo_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + subvideo_length)
            sub, _ = flow_complete.forward_bidirect_flow(
                (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                flow_masks_t[:, s_f:e_f + 1])
            sub = flow_complete.combine_flow(
                (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                sub, flow_masks_t[:, s_f:e_f + 1])
            pred_flows_f.append(sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
            pred_flows_b.append(sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
            torch.cuda.empty_cache()
        pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
    else:
        pred_flows_bi, _ = flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
        pred_flows_bi = flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
        torch.cuda.empty_cache()

    # ---- image propagation ----
    masked_frames = frames_t * (1 - masks_dilated_t)
    subvideo_length_img_prop = min(100, subvideo_length)
    if video_length > subvideo_length_img_prop:
        updated_frames, updated_masks = [], []
        pad_len = 10
        for f in range(0, video_length, subvideo_length_img_prop):
            s_f = max(0, f - pad_len)
            e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

            b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
            sub_flows = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
            prop_imgs, updated_local_masks = generator.img_propagation(
                masked_frames[:, s_f:e_f], sub_flows,
                masks_dilated_t[:, s_f:e_f], 'nearest')
            updated_frames_sub = (
                frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f])
                + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t[:, s_f:e_f])
            updated_masks_sub = updated_local_masks.view(b, t, 1, h, w)

            updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
            updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
            torch.cuda.empty_cache()
        updated_frames = torch.cat(updated_frames, dim=1)
        updated_masks = torch.cat(updated_masks, dim=1)
    else:
        b, t, _, _, _ = masks_dilated_t.size()
        prop_imgs, updated_local_masks = generator.img_propagation(
            masked_frames, pred_flows_bi, masks_dilated_t, 'nearest')
        updated_frames = frames_t * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        torch.cuda.empty_cache()

    # ---- feature propagation + transformer ----
    comp_frames = [None] * video_length
    neighbor_stride = neighbor_length // 2
    ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1

    for f in tqdm(range(0, video_length, neighbor_stride), desc="ProPainter"):
        neighbor_ids = list(range(max(0, f - neighbor_stride),
                                  min(video_length, f + neighbor_stride + 1)))
        ref_ids = _get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
        selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1]],
                                  pred_flows_bi[1][:, neighbor_ids[:-1]])

        l_t = len(neighbor_ids)
        pred_img = generator(selected_imgs, selected_pred_flows_bi,
                             selected_masks, selected_update_masks, l_t)
        pred_img = pred_img.view(-1, 3, h, w)
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
        binary_masks = masks_dilated_t[0, neighbor_ids].cpu().permute(
            0, 2, 3, 1).numpy().astype(np.uint8)

        for i, idx in enumerate(neighbor_ids):
            img = (np.array(pred_img[i]).astype(np.uint8) * binary_masks[i]
                   + ori_frames[idx] * (1 - binary_masks[i]))
            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5
                                    + img.astype(np.float32) * 0.5)
            comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        torch.cuda.empty_cache()

    # ---- back to the original resolution, keeping unmasked pixels intact ----
    results = []
    for idx in range(video_length):
        comp = comp_frames[idx]
        if comp.shape[:2][::-1] != out_size:
            comp = cv2.resize(comp, out_size, interpolation=cv2.INTER_CUBIC)
        orig = np.array(frames[idx].convert("RGB"))
        m = np.array(masks[idx].convert("L").resize(out_size, Image.NEAREST))
        m = (m > 25).astype(np.uint8)[:, :, None]
        results.append(Image.fromarray(np.uint8(comp * m + orig * (1 - m))))
    return results


@torch.no_grad()
def inpaint_video_with_propainter(video_p, mask_dir, output_dir, ckpt_p=None,
                                  fp16=False):
    import imageio.v2 as iio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_propainter_model(ckpt_p, device)

    frames = [Image.fromarray(f) for f in iio.mimread(video_p, memtest=False)]
    mask_ps = sorted(Path(mask_dir).glob("*"))
    masks = [Image.open(p) for p in mask_ps]
    if len(masks) == 1:
        masks = masks * len(frames)

    comp_frames = inpaint_video_with_builded_propainter(
        model, frames, masks, device=device, fp16=fp16)

    video_stem = Path(video_p).stem
    output_p = Path(output_dir) / video_stem / "removed_w_mask.mp4"
    output_p.parent.mkdir(exist_ok=True, parents=True)
    import imageio
    fps = imageio.v3.immeta(video_p, exclude_applied=False).get("fps", 25)
    iio.mimwrite(output_p, [np.array(f) for f in comp_frames], fps=fps)
    print(output_p)


def setup_args(parser):
    parser.add_argument("-v", "--video_p", type=str, required=True)
    parser.add_argument("-m", "--mask_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-c", "--ckpt_p", type=str,
                        default=DEFAULT_PROPAINTER_CKPT_DIR)
    parser.add_argument("--fp16", action="store_true")


if __name__ == "__main__":
    """Example usage:
        python propainter_video_inpaint.py \
            --video_p ./example/video/paragliding/original_video.mp4 \
            --mask_dir ./results/mask_15 \
            --output_dir ./results
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    inpaint_video_with_propainter(
        args.video_p, args.mask_dir, args.output_dir, args.ckpt_p, args.fp16)
