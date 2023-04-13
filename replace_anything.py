from utils import load_img_to_array, save_array_to_img
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from utils.crop_for_replacing import recover_size, resize_and_pad


def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        step: int = 50,
        device="cuda"
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
    ).images[0]
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized


def paints():
    name = "paints"
    seed = torch.seed()
    torch.manual_seed(seed)

    step = 50
    img_p = f"~RI_original/{name}.png"
    img_mask_p = f"~RI_original/{name}/mask.png"
    text_prompt = "painting on an easel in a classroom"
    img_replaced_p = f"~RI_original/{name}/{'_'.join(text_prompt.split(' '))}_step{step}_seed{seed}.png"

    img = load_img_to_array(img_p)
    img_mask = load_img_to_array(img_mask_p)
    img_replaced = replace_img_with_sd(img, img_mask, text_prompt, step)
    save_array_to_img(img_replaced, img_replaced_p)
    print(seed)


def bus():
    name = "bus"
    seed = torch.seed()
    # seed = 7190234422448023767
    torch.manual_seed(seed)

    step = 50
    img_p = f"~RI_original/{name}.jpeg"
    img_mask_p = f"~RI_original/{name}/mask_1.png"
    text_prompt = "bus in Paris street"
    img_replaced_p = f"~RI_original/{name}/{'_'.join(text_prompt.split(' '))}_step{step}_seed{seed}.png"

    img = load_img_to_array(img_p)
    img_mask = load_img_to_array(img_mask_p)
    img_replaced = replace_img_with_sd(img, img_mask, text_prompt, step)
    save_array_to_img(img_replaced, img_replaced_p)
    print(seed)


def dog(index):
    name = "dog"
    seed = torch.seed()
    # seed = 7190234422448023767
    torch.manual_seed(seed)

    step = 50
    img_p = f"~RI_original/{name}.png"
    img_mask_p = f"~RI_original/{name}/mask_1.png"
    text_prompt = "sit on the swing"
    # text_prompt = "on a swing"
    img_replaced_p = f"~RI_original/{name}/{'_'.join(text_prompt.split(' '))}_step{step}_{index}_seed{seed}.png"

    img = load_img_to_array(img_p)
    img_mask = load_img_to_array(img_mask_p)
    img_replaced = replace_img_with_sd(img, img_mask, text_prompt, step)
    save_array_to_img(img_replaced, img_replaced_p)
    print(seed)


