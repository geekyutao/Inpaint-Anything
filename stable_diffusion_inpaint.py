import os
import sys
import glob
import argparse
import torch
import numpy as np
import PIL.Image as Image
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post
from utils.crop_for_replacing import recover_size, resize_and_pad
from utils import load_img_to_array, save_array_to_img


# stabilityai/stable-diffusion-2-inpainting, used by the original release, is no
# longer downloadable. Override with --sd_model or the SD_INPAINT_MODEL env var.
SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
FLUX_FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
SD15_INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

DEFAULT_SD_INPAINT_MODEL = os.environ.get("SD_INPAINT_MODEL", SDXL_INPAINT_MODEL)

# Shorthands accepted by --sd_model.
MODEL_ALIASES = {
    "sdxl": SDXL_INPAINT_MODEL,
    "flux": FLUX_FILL_MODEL,
    "sd15": SD15_INPAINT_MODEL,
}

_PIPE_CACHE = {}


def resolve_model_id(model_id):
    if not model_id:
        return DEFAULT_SD_INPAINT_MODEL
    return MODEL_ALIASES.get(model_id.lower(), model_id)


def model_kind(model_id):
    """Which diffusers pipeline family a model id belongs to."""
    lowered = model_id.lower()
    if "flux" in lowered:
        return "flux"
    if "xl" in lowered:
        return "sdxl"
    return "sd"


def native_crop_size(model_id):
    """Resolution the model was trained at, used to size the crop window."""
    return 512 if model_kind(model_id) == "sd" else 1024


def _load_pipe(model_id, device):
    kind = model_kind(model_id)
    if kind == "flux":
        from diffusers import FluxFillPipeline
        pipe = FluxFillPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16)
        # 12B params do not fit alongside everything else on a 24GB card.
        pipe.enable_model_cpu_offload()
        return pipe
    if kind == "sdxl":
        from diffusers import StableDiffusionXLInpaintPipeline
        return StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16").to(device)
    # The safety checker is skipped: it is not meaningful for a local editing
    # tool, and its weights fail to load on transformers>=5.
    return StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)


def build_sd_inpaint_pipe(model_id=None, device="cuda"):
    """Load the inpainting pipeline once and reuse it across masks."""
    model_id = resolve_model_id(model_id)
    key = (model_id, str(device))
    if key not in _PIPE_CACHE:
        try:
            _PIPE_CACHE[key] = _load_pipe(model_id, device)
        except Exception as e:
            if model_kind(model_id) == "flux":
                raise RuntimeError(
                    f"Failed to load {model_id} ({e}).\n"
                    f"FLUX.1-Fill-dev is a gated HuggingFace repo: accept the "
                    f"license at https://huggingface.co/{FLUX_FILL_MODEL} and "
                    f"run 'hf auth login' first."
                ) from e
            raise
    return _PIPE_CACHE[key]


def _run_pipe(pipe, model_id, prompt, image, mask_image, steps=None):
    """Call the pipeline with the arguments its family expects."""
    from sam3_utils import no_autocast
    with no_autocast():
        return _run_pipe_inner(pipe, model_id, prompt, image, mask_image, steps)


def _run_pipe_inner(pipe, model_id, prompt, image, mask_image, steps=None):
    kind = model_kind(model_id)
    if kind == "flux":
        return pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=image.height,
            width=image.width,
            guidance_scale=30.0,
            num_inference_steps=steps or 50,
            max_sequence_length=512,
        ).images[0]
    kwargs = {}
    if steps is not None:
        kwargs["num_inference_steps"] = steps
    return pipe(prompt=prompt, image=image, mask_image=mask_image,
                **kwargs).images[0]


def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        device="cuda",
        model_id=None
):
    model_id = resolve_model_id(model_id)
    pipe = build_sd_inpaint_pipe(model_id, device)
    crop_size = native_crop_size(model_id)
    img_crop, mask_crop = crop_for_filling_pre(img, mask, crop_size=crop_size)
    img_crop_filled = _run_pipe(
        pipe, model_id, text_prompt,
        Image.fromarray(img_crop), Image.fromarray(mask_crop))
    img_filled = crop_for_filling_post(
        img, mask, np.array(img_crop_filled), crop_size=crop_size)
    return img_filled


def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        step: int = 50,
        device="cuda",
        model_id=None
):
    model_id = resolve_model_id(model_id)
    pipe = build_sd_inpaint_pipe(model_id, device)
    target_size = native_crop_size(model_id)
    img_padded, mask_padded, padding_factors = resize_and_pad(
        img, mask, target_size=target_size)
    # The mask is inverted here: the object is kept and its surroundings are
    # regenerated from the prompt.
    img_padded = _run_pipe(
        pipe, model_id, text_prompt,
        Image.fromarray(img_padded), Image.fromarray(255 - mask_padded),
        steps=step)
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )

if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --text_prompt "a teddy bear on a bench" \
        --output_dir results 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    img_stem = Path(args.input_img).stem
    mask_ps = sorted(glob.glob(args.input_mask_glob))
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_img_to_array(args.input_img)
    for mask_p in mask_ps:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask = load_img_to_array(mask_p)
        img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
        img_filled = fill_img_with_sd(
            img, mask, args.text_prompt, device=device)
        save_array_to_img(img_filled, img_filled_p)