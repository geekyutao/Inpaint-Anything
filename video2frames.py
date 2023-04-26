import torch
from tracking_lib.test.evaluation.video2seq import video2seq


if __name__ == '__main__':
    video_path = './example/remove-anything-video/ikun.mp4'
    coordinates = [290, 341]
    num_points = 1
    sam_ckpt_path = '/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'

    seq, fps = video2seq(
        video_path, 
        coordinates, 
        [num_points], 
        "vit_h", 
        sam_ckpt_path,
        output_dir)