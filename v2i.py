import torch
from tracking_lib.test.evaluation.video2seq import video2seq

video_seq, fps = video2seq(
    '/data1/yutao/projects/Inpaint-Anything/example/remove-anything-video/ikun.mp4', 
    [290, 341], 
    [1], 
    "vit_h", 
    '/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth', 
    './results')

print(video_seq.ground_truth_rect, fps)
