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
    return tracker

def get_box_using_ostrack(tracker, seq, output_dir=None):
    output = tracker.run_sequence(seq, debug=False)
    tracked_bb = np.array(output['target_bbox']).astype(int)
    return tracked_bb



if __name__ == '__main__':
    video_path = './example/remove-anything-video/ikun.mp4'
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

    tracker = Tracker('ostrack', tracker_param, "inpaint-videos")

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    output = tracker.run_sequence(seq, debug=False)
    tracked_bb = np.array(output['target_bbox']).astype(int)
    trajectory_file = osp.join(output_dir, seq.name, 'trajectory.txt')
    np.savetxt(trajectory_file, tracked_bb, delimiter='\t', fmt='%d')

    # # vis frames
    frames_list = vis_traj(seq, output['target_bbox'])
    vis_dir = osp.join(output_dir, seq.name, 'vis_bboxes')
    if not osp.exists(vis_dir):
        os.mkdir(vis_dir)
    for idx, frame in enumerate(frames_list):
        cv2.imwrite(osp.join(vis_dir, '{:05d}.jpg'.format(idx)), frame)

# def video_inpaint(seq: Sequence, tracker: Tracker, inpaint_func=None):
#     print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

#     output, inpainted_frames = tracker.run_video_inpaint(seq, debug=False, inpaint_func=inpaint_func)

#     sys.stdout.flush()

#     return inpainted_frames




    # def inpaint_handler(prompt_bbox, image):
    #     # function to perform frame-wise inpaint
    #     return image
    # inpainted_frames = video_inpaint(video_seq, tracker)
    # frames2video(inpainted_frames, f'{args.output_dir}/{video_seq.name}_inpainted.mp4', fps)
    # shutil.rmtree('./frames')




    # frames = video_seq.frames
    # print(frames)
    # frame_i = frames[5]
    # import cv2
    # from skimage.io import imsave
    # print(frame_i)

    # # Load the image into frame_i
    # frame_i = cv2.imread('image.jpg')

    # # Check the type and shape of frame_i
    # print(type(frame_i), frame_i.shape)

    # # Convert from BGR to RGB color format if necessary
    # if frame_i.ndim == 3 and frame_i.shape[2] == 3:
    #     frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)

    # # Save the converted image
    # imsave('test5.jpg', frame_i)


    # print(video_seq.ground_truth_rect, fps)