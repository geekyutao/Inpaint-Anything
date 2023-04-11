import torch
import numpy as np
import PIL.Image as Image
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    # torch_dtype=torch.float16,
    torch_dtype=torch.float32,
)
pipe.to("cuda")

prompt = "A fox, waiting for food"
img_path = '/data1/yutao/projects/IAM/example/remove-anything/dog.jpg'
mask_path = '/data1/yutao/projects/IAM/example/remove-anything/dog_mask.png'

image = Image.open(img_path).resize((512, 512))
mask_image = Image.open(mask_path).resize((512, 512))

#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./fox.png")
