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
from sam_segment import build_sam_model, ALL_MODEL_TYPES, DEFAULT_SAM3_CKPT
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
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
        "--sam_model_type", type=str,
        default="sam3", choices=list(ALL_MODEL_TYPES),
        help="The type of sam model to load. Default: 'sam3'",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default=DEFAULT_SAM3_CKPT,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the Gradio server on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
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

def _restore_sam_state(features, orig_h, orig_w, input_h, input_w):
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w


def _apply_dilate(masks, dilate_kernel_size):
    if not dilate_kernel_size:
        return [mask for mask in masks]
    return [dilate_mask(mask, int(dilate_kernel_size)) for mask in masks]


def get_click_mask(clicked_points, features, orig_h, orig_w, input_h, input_w,
                   dilate_kernel_size):
    _restore_sam_state(features, orig_h, orig_w, input_h, input_w)

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
    return _apply_dilate(masks, dilate_kernel_size)

def process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, evt: gr.SelectData):
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
    mask_click_np = get_click_mask(resized_points, features, orig_h, orig_w,
                                   input_h, input_w, dilate_kernel_size)

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

def process_text_select(original_image, image_resolution, text_select,
                        text_confidence, dilate_kernel_size,
                        features, orig_h, orig_w, input_h, input_w):
    if original_image is None:
        raise gr.Error("Please upload an image first.")
    if not text_select or not text_select.strip():
        raise gr.Error("Please enter a noun phrase, e.g. 'dog'.")
    if not hasattr(model['sam'], "predict_text"):
        raise gr.Error("Text selection requires --sam_model_type sam3.")

    input_image = np.array(original_image, dtype=np.uint8)
    H, W, C = input_image.shape
    input_image = HWC3(input_image)

    # Reuse the backbone output cached on upload instead of re-encoding.
    _restore_sam_state(features, orig_h, orig_w, input_h, input_w)
    masks, _, _ = model['sam'].predict_text(
        text_select.strip(), confidence_threshold=text_confidence)

    if len(masks) == 0:
        raise gr.Error(
            f"No object matched '{text_select}'. Try a simpler phrase or "
            f"lower the confidence threshold."
        )

    merged = np.any(masks, axis=0).astype(np.uint8)
    merged = _apply_dilate([merged], dilate_kernel_size)[0]

    mask_image = HWC3((merged[:, :, None] * 255.0).astype(np.uint8))
    mask_image = cv2.resize(mask_image, (W, H), interpolation=cv2.INTER_LINEAR)

    overlay_image = cv2.addWeighted(
        input_image,
        1.0,
        (mask_image * np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        0.75,
        0,
    )
    return overlay_image, mask_image


def _get_video_models(fp16):
    """Build the video tracker and inpainter on first use and keep them around."""
    if 'sam3_video' not in model:
        from sam3_video_track import build_sam3_video_tracker
        model['sam3_video'] = build_sam3_video_tracker(args.sam_ckpt, device=device)
    if 'propainter' not in model:
        from propainter_video_inpaint import build_propainter_model
        model['propainter'] = build_propainter_model(device=device)
    return model['sam3_video'], model['propainter']


def video_first_frame(video_path):
    """Returns (preview, cleared clicks, pristine first frame)."""
    if not video_path:
        return None, [], None
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise gr.Error("Could not read the first frame of that video.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, [], rgb


def video_frame_click(first_frame, clicked_points, point_prompt,
                      evt: gr.SelectData):
    """Preview the first-frame selection before running the whole video."""
    if first_frame is None:
        raise gr.Error("Please upload a video first.")
    x, y = evt.index
    lab = 1 if point_prompt == "Foreground Point" else 0
    clicked_points = (clicked_points or []) + [(x, y, lab)]

    pts = np.array([[p[0], p[1]] for p in clicked_points])
    labels = np.array([p[2] for p in clicked_points])
    model['sam'].set_image(first_frame)
    masks, scores, _ = model['sam'].predict(
        point_coords=pts, point_labels=labels, multimask_output=False)
    model['sam'].reset_image()

    mask = (masks[0] > 0).astype(np.uint8)
    overlay = _overlay_mask(first_frame, mask)
    for px, py, plab in clicked_points:
        overlay = cv2.circle(overlay, (px, py), 10,
                             (255, 0, 0) if plab == 1 else (0, 0, 255), -1)
    return overlay, clicked_points, mask * 255


def _overlay_mask(img, mask):
    mask_rgb = HWC3((mask[:, :, None] * 255).astype(np.uint8))
    return cv2.addWeighted(
        img.astype(np.uint8), 1.0,
        (mask_rgb * np.array([0, 1, 0])).astype(np.uint8), 0.6, 0)


def video_remove(video_path, first_frame, clicked_points, text_select,
                 text_confidence, dilate_kernel_size, fp16,
                 progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    if not text_select and not clicked_points:
        raise gr.Error("Click the object in the first frame, or type a phrase.")

    from sam3_video_track import track_masks_in_video
    from propainter_video_inpaint import inpaint_video_with_builded_propainter
    import imageio.v2 as iio
    import imageio

    progress(0.05, desc="Loading models")
    tracker, painter = _get_video_models(fp16)

    progress(0.15, desc="Selecting the object in frame 0")
    seed = {}
    if text_select:
        if not hasattr(model['sam'], "predict_text"):
            raise gr.Error("Text selection requires --sam_model_type sam3.")
        model['sam'].set_image(first_frame)
        masks, _, _ = model['sam'].predict_text(
            text_select.strip(), confidence_threshold=text_confidence)
        model['sam'].reset_image()
        if len(masks) == 0:
            raise gr.Error(f"No object matched '{text_select}'.")
        seed["init_mask"] = masks[0]
    else:
        pts = np.array([[p[0], p[1]] for p in clicked_points])
        labels = np.array([p[2] for p in clicked_points])
        model['sam'].set_image(first_frame)
        masks, _, _ = model['sam'].predict(
            point_coords=pts, point_labels=labels, multimask_output=False)
        model['sam'].reset_image()
        seed["init_mask"] = masks[0] > 0

    progress(0.3, desc="Tracking through the video")
    raw_masks = track_masks_in_video(tracker, video_path, **seed)

    frames = [Image.fromarray(f) for f in iio.mimread(video_path, memtest=False)]
    frames = frames[:len(raw_masks)]
    pil_masks = []
    for m in raw_masks[:len(frames)]:
        m = m.astype(np.uint8)
        if dilate_kernel_size:
            m = dilate_mask(m, int(dilate_kernel_size))
        pil_masks.append(Image.fromarray(np.uint8(m * 255)))

    progress(0.5, desc="Inpainting with ProPainter")
    out_frames = inpaint_video_with_builded_propainter(
        painter, frames, pil_masks, device=device, fp16=fp16)

    progress(0.95, desc="Encoding")
    try:
        fps = imageio.v3.immeta(video_path, exclude_applied=False).get("fps", 25)
    except Exception:
        fps = 25
    out_p = str(mkstemp(".mp4"))
    iio.mimwrite(out_p, [np.array(f) for f in out_frames], fps=fps)

    mask_p = str(mkstemp(".mp4"))
    iio.mimwrite(mask_p, [np.array(m.convert("RGB")) for m in pil_masks], fps=fps)
    return out_p, mask_p


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
device = "cuda" if torch.cuda.is_available() else "cpu"
model['sam'] = build_sam_model(args.sam_model_type, args.sam_ckpt, device=device)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks(title="Inpaint Anything") as demo:
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    click_mask = gr.State(None)
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    gr.Markdown(
        "# Inpaint Anything\n"
        "Segmentation by **SAM 3** · image inpainting by **LaMa / Stable Diffusion** "
        "· video inpainting by **ProPainter**"
    )

    with gr.Tabs():
        with gr.Tab("Image"):
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
                    with gr.Row():
                        text_select = gr.Textbox(
                            label="Select by Text (SAM 3)",
                            placeholder="e.g. dog — selects every match, no clicking needed",
                        )
                        text_confidence = gr.Slider(
                            label="Text Confidence", minimum=0.05, maximum=0.95,
                            step=0.05, value=0.5,
                        )
                    sam_text = gr.Button("Segment by Text", variant="secondary")
                    text_prompt = gr.Textbox(label="Text Prompt (what to generate)")
                    lama = gr.Button("Inpaint Image", variant="primary")
                    replace_sd = gr.Button("Replace Anything with SD", variant="primary")
                    clear_button_image = gr.Button(value="Reset", variant="secondary")

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

        with gr.Tab("Video"):
            v_first_frame = gr.State(None)
            v_points = gr.State([])

            gr.Markdown(
                "Upload a video, pick the object in the first frame (click it or name "
                "it), then remove it from every frame. Tracking uses SAM 3, inpainting "
                "uses ProPainter."
            )
            with gr.Row():
                with gr.Column(variant="panel"):
                    v_input = gr.Video(label="Input video", height=300)
                    v_frame = gr.Image(type="numpy", height=300, interactive=True,
                                       label="First frame — click the object to remove")
                    v_point_prompt = gr.Radio(
                        choices=["Foreground Point", "Background Point"],
                        value="Foreground Point", show_label=False, interactive=True)
                with gr.Column(variant="panel"):
                    v_text = gr.Textbox(
                        label="Or select by text (SAM 3)",
                        placeholder="e.g. paraglider")
                    v_text_conf = gr.Slider(label="Text Confidence", minimum=0.05,
                                            maximum=0.95, step=0.05, value=0.5)
                    v_dilate = gr.Slider(label="Dilate Kernel Size", minimum=0,
                                         maximum=30, step=1, value=8)
                    v_fp16 = gr.Checkbox(
                        label="Half precision (less GPU memory, slightly lower quality)",
                        value=False)
                    v_run = gr.Button("Remove from Video", variant="primary")
                    v_mask_preview = gr.Image(type="numpy", label="First-frame mask")
            with gr.Row(variant="panel"):
                v_output = gr.Video(label="Object removed")
                v_mask_video = gr.Video(label="Tracked mask")

            v_input.change(video_first_frame, inputs=[v_input],
                           outputs=[v_frame, v_points, v_first_frame])
            v_frame.select(video_frame_click,
                           inputs=[v_first_frame, v_points, v_point_prompt],
                           outputs=[v_frame, v_points, v_mask_preview])
            v_run.click(
                video_remove,
                inputs=[v_input, v_first_frame, v_points, v_text, v_text_conf,
                        v_dilate, v_fp16],
                outputs=[v_output, v_mask_video],
            )

    # --- event wiring ---
    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[origin_image, features, orig_h, orig_w, input_h, input_w],
    )
    source_image_click.select(
        process_image_click,
        inputs=[origin_image, point_prompt,
                clicked_points, image_resolution,
                features, orig_h, orig_w, input_h, input_w,
                dilate_kernel_size],
        outputs=[source_image_click, clicked_points, click_mask],
        show_progress=True,
    )

    sam_text.click(
        process_text_select,
        inputs=[origin_image, image_resolution, text_select, text_confidence,
                dilate_kernel_size,
                features, orig_h, orig_w, input_h, input_w],
        outputs=[source_image_click, click_mask],
        show_progress=True,
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
    demo.queue(api_open=False).launch(
        server_name='0.0.0.0', server_port=args.port, share=args.share,
        debug=True)
