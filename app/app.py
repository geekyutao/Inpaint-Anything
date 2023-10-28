import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
# from omegaconf import OmegaConf
# from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse

def setup_args(parser):
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

def get_replace_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    H, W, C = np_image.shape
    np_image = HWC3(np_image)
    np_image = resize_image(np_image, image_resolution)

    img_replaced = replace_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_points(clicked_points, original_shape, resolution):
    original_height, original_width, _ = original_shape
    original_height = float(original_height)
    original_width = float(original_width)
    
    scale_factor = float(resolution) / min(original_height, original_width)
    resized_points = []
    
    for point in clicked_points:
        x, y, lab = point
        resized_x = int(round(x * scale_factor))
        resized_y = int(round(y * scale_factor))
        resized_point = (resized_x, resized_y, lab)
        resized_points.append(resized_point)
    
    return resized_points

def get_click_mask(clicked_points, features, orig_h, orig_w, input_h, input_w):
    # model['sam'].set_image(image)
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    
    # Separate the points and labels
    points, labels = zip(*[(point[:2], point[2])
                            for point in clicked_points])

    # Convert the points and labels to numpy arrays
    input_point = np.array(points)
    input_label = np.array(labels)

    masks, _, _ = model['sam'].predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size.value) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

def process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, evt: gr.SelectData):
    clicked_coords = evt.index
    x, y = clicked_coords
    label = point_prompt
    lab = 1 if label == "Foreground Point" else 0
    clicked_points.append((x, y, lab))

    input_image = np.array(original_image, dtype=np.uint8)
    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)

    # Update the clicked_points
    resized_points = resize_points(
        clicked_points, input_image.shape, image_resolution
    )
    mask_click_np = get_click_mask(resized_points, features, orig_h, orig_w, input_h, input_w)

    # Convert mask_click_np to HWC format
    mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0

    mask_image = HWC3(mask_click_np.astype(np.uint8))
    mask_image = cv2.resize(
        mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
    # mask_image = Image.fromarray(mask_image_tmp)

    # Draw circles for all clicked points
    edited_image = input_image
    for x, y, lab in clicked_points:
        # Set the circle color based on the label
        color = (255, 0, 0) if lab == 1 else (0, 0, 255)

        # Draw the circle
        edited_image = cv2.circle(edited_image, (x, y), 20, color, -1)

    # Set the opacity for the mask_image and edited_image
    opacity_mask = 0.75
    opacity_edited = 1.0

    # Combine the edited_image and the mask_image using cv2.addWeighted()
    overlay_image = cv2.addWeighted(
        edited_image,
        opacity_edited,
        (mask_image *
            np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        opacity_mask,
        0,
    )

    return (
        overlay_image,
        # Image.fromarray(overlay_image),
        clicked_points,
        # Image.fromarray(mask_image),
        mask_image
    )

def image_upload(image, image_resolution):
    if image is not None:
        np_image = np.array(image, dtype=np.uint8)
        H, W, C = np_image.shape
        np_image = HWC3(np_image)
        np_image = resize_image(np_image, image_resolution)
        features, orig_h, orig_w, input_h, input_w = get_sam_feat(np_image)
        return image, features, orig_h, orig_w, input_h, input_w
    else:
        return None, None, None, None, None, None

def get_inpainted_img(image, mask, image_resolution):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image, mask, lama_config, device=device)
    return img_inpainted


# get args 
parser = argparse.ArgumentParser()
setup_args(parser)
args = parser.parse_args(sys.argv[1:])
# build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p=args.sam_ckpt
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks() as demo:
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    click_mask = gr.State(None)
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image")
            with gr.Row():
                # img = gr.Image(label="Input Image")
                source_image_click = gr.Image(
                    type="numpy",
                    height=300,
                    interactive=True,
                    label="Image: Upload an image and click the region you want to edit.",
                )
            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Foreground Point",
                                "Background Point"],
                    value="Foreground Point",
                    label="Point Label",
                    interactive=True,
                    show_label=False,
                )
                image_resolution = gr.Slider(
                    label="Image Resolution",
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                )
                dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=30, step=1, value=3)
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            text_prompt = gr.Textbox(label="Text Prompt")
            lama = gr.Button("Inpaint Image", variant="primary")
            replace_sd = gr.Button("Replace Anything with SD", variant="primary")
            clear_button_image = gr.Button(value="Reset", label="Reset", variant="secondary")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Mask")
            with gr.Row():
                click_mask = gr.Image(type="numpy", label="Click Mask")
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Removed with Mask")
            with gr.Row():
                img_rm_with_mask = gr.Image(
                    type="numpy", label="Image Removed with Mask")
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Replace Anything with Mask")
            with gr.Row():
                img_replace_with_mask = gr.Image(
                    type="numpy", label="Image Replace Anything with Mask")

    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[origin_image, features, orig_h, orig_w, input_h, input_w],
    )
    source_image_click.select(
        process_image_click,
        inputs=[origin_image, point_prompt,
                clicked_points, image_resolution,
                features, orig_h, orig_w, input_h, input_w],
        outputs=[source_image_click, clicked_points, click_mask],
        show_progress=True,
        queue=True,
    )

    # sam_mask.click(
    #     get_masked_img,
    #     [origin_image, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
    #     [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
    # )

    lama.click(
        get_inpainted_img,
        [origin_image, click_mask, image_resolution],
        [img_rm_with_mask]
    )
    
    replace_sd.click(
        get_replace_img_with_sd,
        [origin_image, click_mask, image_resolution, text_prompt],
        [img_replace_with_mask]
    )


    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        [origin_image, features, click_mask, img_rm_with_mask, img_replace_with_mask],
        [origin_image, features, click_mask, img_rm_with_mask, img_replace_with_mask]
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(server_name='0.0.0.0', share=False, debug=True)