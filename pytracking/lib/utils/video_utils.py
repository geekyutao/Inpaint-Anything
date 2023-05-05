import cv2
import imageio 
import os 
import shutil
from tqdm import tqdm
from glob import glob 
from os import path as osp

def video2frames(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    initial_img = None
    for idx in tqdm(range(frame_num), 'Extract frames'):
        success, image = video.read()
        if idx == 0: initial_img = image.copy()
        assert success, 'extract the {}th frame in video {} failed!'.format(idx, video_path)
        cv2.imwrite("{}/{:05d}.jpg".format(frame_path, idx), image)
    return fps, initial_img

def frames2video(frames_list, video_path, fps=30, remove_tmp=False):
    # frames_list: frames dir or list of images.
    if isinstance(frames_list, str):
        frames_list = glob(f'{frames_list}/*.jpg')
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in tqdm(frames_list, 'Export video'):
        if isinstance(frame, str):
            frame = imageio.imread(frame)
        else:
            # convert cv2 (rgb) to PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imageio.core.util.Array(frame)
        writer.append_data(frame)
    writer.close()
    print(f'find video at {video_path}.')
    if remove_tmp and isinstance(frames_list, str):
        shutil.rmtree(frames_list)

if __name__ == '__main__':
    fps = video2frames('./unitest/example.mp4', './unitest/frames/')
    frames2video('./unitest/frames/', './unitest/new.mp4', fps, True)
