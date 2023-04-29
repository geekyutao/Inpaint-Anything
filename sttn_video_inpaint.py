import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parent / "sttn"))
from core.utils import Stack, ToTorchFormatTensor


_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()]
)


def get_ref_index(neighbor_ids, length):
    ref_length = 10
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames:
        m = Image.open(os.path.join(mpath, m))
        # m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # frames.append(image.resize((w, h)))
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames


def build_sttn_model(ckpt_p, model_type="sttn", device="cuda"):
    net = importlib.import_module(f'model.{model_type}')
    model = net.InpaintGenerator().to(device)
    data = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(data['netG'])
    model.eval()
    return model


@torch.no_grad()
def inpaint_video_with_builded_sttn(
        model,
        frames: List[Image.Image],
        masks: List[Image.Image],
        device="cuda"
) -> List[Image.Image]:
    w, h = 432, 240
    neighbor_stride = 5
    video_length = len(frames)

    feats = [frame.resize((w, h)) for frame in frames]
    feats = _to_tensors(feats).unsqueeze(0) * 2 - 1
    _masks = [mask.resize((w, h), Image.NEAREST) for mask in masks]
    _masks = _to_tensors(_masks).unsqueeze(0)

    feats, _masks = feats.to(device), _masks.to(device)
    comp_frames = [None] * video_length

    feats = (feats * (1 - _masks).float()).view(video_length, 3, h, w)
    feats = model.encoder(feats)
    _, c, feat_h, feat_w = feats.size()
    feats = feats.view(1, video_length, c, feat_h, feat_w)

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = list(range(max(0, f - neighbor_stride),
                                  min(video_length, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(neighbor_ids, video_length)

        pred_feat = model.infer(feats[0, neighbor_ids + ref_ids, :, :, :],
                                _masks[0, neighbor_ids + ref_ids, :, :, :])
        pred_img = model.decoder(pred_feat[:len(neighbor_ids), :, :, :])
        pred_img = torch.tanh(pred_img)
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.permute(0, 2, 3, 1) * 255
        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            b_mask = _masks.squeeze()[idx].unsqueeze(-1)
            b_mask = (b_mask != 0).int()
            frame = torch.from_numpy(np.array(frames[idx].resize((w, h))))
            frame = frame.to(device)
            img = pred_img[i] * b_mask + frame * (1 - b_mask)
            img = img.cpu().numpy()
            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx] * 0.5 + img * 0.5

    ori_w, ori_h = frames[0].size
    for idx in range(len(frames)):
        frame = np.array(frames[idx])
        b_mask = np.uint8(np.array(masks[idx])[..., np.newaxis] != 0)
        comp_frame = np.uint8(comp_frames[idx])
        comp_frame = Image.fromarray(comp_frame).resize((ori_w, ori_h))
        comp_frame = np.array(comp_frame)
        comp_frame = comp_frame * b_mask + frame * (1 - b_mask)
        comp_frames[idx] = Image.fromarray(np.uint8(comp_frame))
    return comp_frames

@torch.no_grad()
def inpaint_video_with_sttn(
        video_p,
        mask_dir,
        output_dir,
        ckpt_p,
        model_type="sttn"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build sttn model
    model = build_sttn_model(ckpt_p, model_type, device)

    # prepare dataset, encode all frames into deep space
    frames = read_frame_from_videos(video_p)
    masks = read_mask(mask_dir)

    # inference
    comp_frames = inpaint_video_with_builded_sttn(
        model, frames, masks, device)

    video_stem = Path(video_p).stem
    output_p = Path(output_dir) / video_stem/ f"removed_w_mask.mp4"
    output_p.parent.mkdir(exist_ok=True, parents=True)

    w, h = frames[0].size
    fps = imageio.v3.immeta(video_p, exclude_applied=False)["fps"]
    writer = cv2.VideoWriter(
        str(output_p),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    for idx in range(len(comp_frames)):
        writer.write(cv2.cvtColor(np.uint8(comp_frames[idx]), cv2.COLOR_BGR2RGB))
    writer.release()
    print(output_p)


def setup_args(parser):
    parser.add_argument("-v", "--video_p", type=str, required=True)
    parser.add_argument("-m", "--mask_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-c", "--ckpt_p", type=str, required=True)
    parser.add_argument("--model", type=str, default='sttn')


if __name__ == '__main__':
    '''
    1. Download STTN pretrained model and move it to ./pretrained_models/sttn.pth
    2. Run:
        python sttn_video_inpaint.py \
            --video_p ./example/remove-anything-video/breakdance-flare/original_video.mp4 \
            --mask_dir ./example/remove-anything-video/breakdance-flare/mask \
            --output_dir ./results
            --ckpt_p pretrained_models/sttn.pth
    '''
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    inpaint_video_with_sttn(
        args.video_p, args.mask_dir, args.output_dir, args.ckpt_p)
