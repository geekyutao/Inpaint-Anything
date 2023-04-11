import os
import cv2
import torch
import numpy as np
import PIL.Image as Image
from diffusers import StableDiffusionInpaintPipeline
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post

output_dir = "/data1/yutao/projects/IAM/select_results"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    # torch_dtype=torch.float16,
    torch_dtype=torch.float32,
)
pipe.to("cuda")

prompt = "A cute cat on a bench, looking straight ahead, high quality"
img_path = '/data1/yutao/projects/IAM/selected_results/FA1_dog/with_points.png'
mask_path = '/data1/yutao/projects/IAM/selected_results/FA1_dog/mask_1.png'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
cropped_image, cropped_mask = crop_for_filling_pre(image, mask)
input_image = Image.fromarray(cropped_image)
input_mask = Image.fromarray(cropped_mask)
output_image = pipe(prompt=prompt, image=input_image, mask_image=input_mask).images[0]
output_image = np.array(output_image)
image = crop_for_filling_post(image, mask, output_image)
image = Image.fromarray(image)
image.save(os.path.splitext(img_path)[0] + '_result.png')


