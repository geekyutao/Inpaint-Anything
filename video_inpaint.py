import torch
import sys
import argparse
import numpy as np
import shutil
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points

from lib.test.evaluation.video2seq import video2seq
from lib.test.evaluation import Tracker
from lib.test.evaluation.running import video_inpaint

from lib.utils.video_utils import frames2video



def setup_args(parser):
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a input video",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt in first frame, [coord_W coord_H].",
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
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )


if __name__ == "__main__":
    """Example usage:
    python video_inpaint.py \
        --input_video example/rabbit.mp4 \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_seq, fps = video2seq(args.input_video, args.point_coords, args.point_labels, args.sam_model_type, args.sam_ckpt)
    tracker_param = 'vitb_256_mae_ce_32x4_ep300.yaml'
    tracker = Tracker('ostrack', tracker_param, "inpaint-videos")
    def inpaint_handler(prompt_bbox, image):
        # function to perform frame-wise inpaint
        return image
    inpainted_frames = video_inpaint(video_seq, tracker)
    frames2video(inpainted_frames, f'{args.output_dir}/{video_seq.name}_inpainted.mp4', fps)
    shutil.rmtree('./frames')
