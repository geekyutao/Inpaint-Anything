import os
import cv2
import torch 
import numpy as np

from pytracking.lib.test.evaluation.data import Sequence
from pytracking.lib.utils.video_utils import video2frames
from sam_segment import predict_masks_with_sam

def video2seq(video_path, point_coords, point_labels, sam_model_type, sam_ckpt, output_dir):
    video_name, _ = os.path.splitext(video_path.split('/')[-1])
    frames_path = f'{output_dir}/{video_name}/original_frames'
    fps, first_frame = video2frames(video_path, frames_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    masks, scores, _ = predict_masks_with_sam(
        first_frame,
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
    # # visualize the intialized gt_bbox
    # vis_img = cv2.cvtColor(first_frame.copy(), cv2.COLOR_RGB2BGR)
    # vis_img = cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    # cv2.imwrite('./gt_bbox.jpg', vis_img)
    gt_rect = np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.float32).reshape(-1, 4)
    frames_list = [os.path.join(frames_path, frame) for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
    return Sequence(video_name, frames_list, 'inpaint-anything', gt_rect.reshape(-1, 4)), fps