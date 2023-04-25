import os
import cv2
import torch 
import numpy as np

from lib.test.evaluation.data import Sequence
from lib.utils.video_utils import video2frames
from sam_segment import predict_masks_with_sam

def video2seq(video_path, point_coords, point_labels, sam_model_type, sam_ckpt):
    video_name, _ = os.path.splitext(video_path.split('/')[-1])
    frames_path = f'./frames/{video_name}'
    fps, initial_img = video2frames(video_path, frames_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB)
    masks, scores, _ = predict_masks_with_sam(
        initial_img,
        [point_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    mask = masks[np.argmax(scores)][:, :, None].astype(np.uint8) * 255
    mask_loc = np.where(mask > 0)
    x1, x2 = np.min(mask_loc[1]), np.max(mask_loc[1])
    y1, y2 = np.min(mask_loc[0]), np.max(mask_loc[0])
    x1, y1, x2, y2 = list(map(lambda x: int(x), [x1, y1, x2, y2]))
    gt_rect = np.array([x1, y1, x2, y2]).astype(np.float32).reshape(-1, 4)
    frames_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
    return Sequence(video_name, frames_list, 'inpaint-anything', gt_rect.reshape(-1, 4)), fps

