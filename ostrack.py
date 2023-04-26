import torch
import numpy as np
import cv2
import os
from os import path as osp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pytracking"))
from pytracking.lib.test.evaluation.video2seq import video2seq
from pytracking.lib.test.evaluation import Tracker
from pytracking.lib.utils.video_utils import frames2video

def vis_traj(seq, output_boxes):
    frames_list = []
    for frame, box in zip(seq.frames, output_boxes):
        frame = cv2.imread(frame)
        x, y, w, h = box
        x1, y1, x2, y2 = map(lambda x: int(x), [x, y, (x + w), (y+h)])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        frames_list.append(frame)
    return frames_list

    # seq.frames[1:]

def build_ostrack_model(tracker_param):
    tracker = Tracker('ostrack', tracker_param, "inpaint-videos")
    # print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))
    return tracker

def get_box_using_ostrack(tracker, seq, output_dir=None):
    output = tracker.run_sequence(seq, debug=False)
    tracked_bb = np.array(output['target_bbox']).astype(int)
    # trajectory_file = osp.join(output_dir, seq.name, 'trajectory.txt')
    # np.savetxt(trajectory_file, tracked_bb, delimiter='\t', fmt='%d')
    return tracked_bb



if __name__ == '__main__':
    video_path = '/data1/yutao/projects/Inpaint-Anything/example/remove-anything-video/ikun.mp4'
    coordinates = [290, 341]
    num_points = 1
    sam_ckpt_path = '/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'
    tracker_param = 'vitb_384_mae_ce_32x4_ep300'

    seq, fps = video2seq(
        video_path, 
        coordinates, 
        [num_points], 
        "vit_h", 
        sam_ckpt_path, 
        output_dir)


    tracker = build_ostrack_model(tracker_param)
    get_box_using_ostrack(tracker, seq, output_dir)