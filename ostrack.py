import torch
from tracking_lib.test.evaluation.video2seq import video2seq
from tracking_lib.test.evaluation import Tracker


if __name__ == '__main__':
    video_path = './example/remove-anything-video/ikun.mp4'
    coordinates = [290, 341]
    num_points = 1
    sam_ckpt_path = '/data1/yutao/projects/IAM/pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'
    # tracker_param = 'vitb_256_mae_ce_32x4_ep300.yaml'
    tracker_param = 'vitb_384_mae_ce_32x4_ep300.yaml'

    seq, fps = video2seq(
        video_path, 
        coordinates, 
        [num_points], 
        "vit_h", 
        sam_ckpt_path, 
        output_dir)

    tracker = Tracker('ostrack', tracker_param, "inpaint-videos")

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    # output, inpainted_frames = tracker.run_video_inpaint(seq, debug=False, inpaint_func=inpaint_func)


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