import cv2
import imageio 
import os 
import shutil
from tqdm import tqdm
from glob import glob 
from os import path as osp

def video2frames(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    if not osp.exists(frame_path):
        os.mkdir(frame_path)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    for idx in tqdm(range(frame_num), 'Extract frames'):
        success, image = video.read()
        assert success, 'extract the {}th frame in video {} failed!'.format(idx, video_path)
        cv2.imwrite("{}/{:05d}.jpg".format(frame_path, idx), image)
    return fps 

def framesvideo(frame_path, video_path, fps=30, remove_tmp=False):
    frames_list = glob(f'{frame_path}/*.jpg')
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in tqdm(frames_list, 'Export video'):
        image = imageio.imread(frame)
        writer.append_data(image)
    writer.close()
    print(f'find video at {video_path}.')
    if remove_tmp:
        shutil.rmtree(frame_path)

if __name__ == '__main__':
    # fps = video2frames('./unitest/example.mp4', './unitest/frames/')
    fps = 25.0
    framesvideo('./unitest/frames/', './unitest/new.mp4', fps, True)
