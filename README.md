<p align="center">
  <img src="./example/IAM.png">
</p>

# Inpaint Anything: Segment Anything Meets Image Inpainting
- Authors: Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng and Zhibo Chen.
- Institutes: University of Science and Technology of China; Eastern Institute for Advanced Study.
- Paper: [arXiv](https://arxiv.org/abs/2304.06790)
<p align="center">
  <img src="./example/MainFramework.png" width="100%">
</p>

TL; DR: Users can select any object in an image by clicking on it. With powerful vision models, e.g., [SAM](https://arxiv.org/abs/2304.02643), [LaMa](https://arxiv.org/abs/2109.07161) and [Stable Diffusion (SD)](https://arxiv.org/abs/2112.10752), **Inpaint Anything** is able to remove the object smoothly (i.e., *Remove Anything*). Further, prompted by user input text, Inpaint Anything can fill the object with any desired content (i.e., *Fill Anything*) or replace the background of it arbitrarily (i.e., *Replace Anything*).


## 🌟 Inpaint Anything Features
- [x] **Remove** Anything
- [x] **Fill** Anything
- [x] **Replace** Anything

## 💡 Highlights
- [x] Any aspect ratio supported
- [x] 2K resolution supported
- [x] [Technical report on arXiv](https://arxiv.org/abs/2304.06790)
- [ ] Demo Website (coming soon)


<!-- ## Updates
| Date | News |
| ------ | --------
| 2023-04-12 | Release the Fill Anything feature | 
| 2023-04-10 | Release the Remove Anything feature |
| 2023-04-10 | Release the first version of Inpaint Anything | -->

## 🔥 Remove Anything


<!-- <table>
  <tr>
    <td><img src="./example/remove-anything/dog/with_points.png" width="100%"></td>
    <td><img src="./example/remove-anything/dog/with_mask.png" width="100%"></td>
    <td><img src="./example/remove-anything/dog/inpainted_with_mask.png" width="100%"></td>
  </tr>
</table> -->

<p align="center">
    <img src="./example/GIF/Remove-dog.gif"  alt="image" style="width:400px;">
</p>


**Click** on an object in the image, and Inpainting Anything will **remove** it instantly!
- Click on an object;
- [Segment Anything Model](https://segment-anything.com/) (SAM) segments the object out;
- Inpainting models (e.g., [LaMa](https://advimman.github.io/lama-project/)) fill the "hole".

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install -r lama/requirements.txt 
```

### Usage
Download the model checkpoints provided in [segment_anything](./segment_anything/README.md) 
and [lama](./lama/README.md) (e.g. [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 
and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)).

Specify an image and a point, and Inpaint-Anything will remove the object at the point.
```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --point_coords 200 450 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt big-lama
```

### Demo
<table>
  <tr>
    <td><img src="./example/remove-anything/person/with_points.png" width="100%"></td>
    <td><img src="./example/remove-anything/person/with_mask.png" width="100%"></td>
    <td><img src="./example/remove-anything/person/inpainted_with_mask.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./example/remove-anything/bridge/with_points.png" width="100%"></td>
    <td><img src="./example/remove-anything/bridge/with_mask.png" width="100%"></td>
    <td><img src="./example/remove-anything/bridge/inpainted_with_mask.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./example/remove-anything/boat/with_points.png" width="100%"></td>
    <td><img src="./example/remove-anything/boat/with_mask.png" width="100%"></td>
    <td><img src="./example/remove-anything/boat/inpainted_with_mask.png" width="100%"></td>
  </tr>
</table>


<table>
  <tr>
    <td><img src="./example/remove-anything/baseball/with_points.png" width="100%"></td>
    <td><img src="./example/remove-anything/baseball/with_mask.png" width="100%"></td>
    <td><img src="./example/remove-anything/baseball/inpainted_with_mask.png" width="100%"></td>
  </tr>
</table>



## 🔥 Fill Anything
<!-- <table>
  <caption align="center">Text prompt: "a teddy bear on a bench"</caption>
    <tr>
      <td><img src="./example/fill-anything/sample1/with_points.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample1/with_mask.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample1/filled_with_mask.png" width="100%"></td>
    </tr>
</table> -->
<p align="center">Text prompt: "a teddy bear on a bench"</p>
<p align="center">
    <img src="./example/GIF/Fill-sample1.gif" alt="image" style="width:400px;">
</p>

**Click** on an object, **type** in what you want to fill, and Inpaint Anything will **fill** it!
- Click on an object;
- [SAM](https://segment-anything.com/) segments the object out;
- Input a text prompt;
- Text-prompt-guided inpainting models (e.g., [Stable Diffusion](https://github.com/CompVis/stable-diffusion)) fill the "hole" according to the text.

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
```

### Usage
Download the model checkpoints provided in [segment_anything](./segment_anything/README.md)
(e.g. [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)).

Specify an image, a point and text prompt, and run:
```bash
python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt sam_vit_h_4b8939.pth
```

### Demo

<table>
  <caption align="center">Text prompt: "a camera lens in the hand"</caption>
    <tr>
      <td><img src="./example/fill-anything/sample2/with_points.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample2/with_mask.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample2/filled_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "a Picasso painting on the wall"</caption>
    <tr>
      <td><img src="./example/fill-anything/sample5/with_points.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample5/with_mask.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample5/filled_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "an aircraft carrier on the sea"</caption>
    <tr>
      <td><img src="./example/fill-anything/sample3/with_points.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample3/with_mask.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample3/filled_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "a sports car on a road"</caption>
    <tr>
      <td><img src="./example/fill-anything/sample4/with_points.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample4/with_mask.png" width="100%"></td>
      <td><img src="./example/fill-anything/sample4/filled_with_mask.png" width="100%"></td>
    </tr>
</table>


## 🔥 Replace Anything
<!-- <table>
  <caption align="center">Text prompt: "a man in office"</caption>
    <tr>
      <td><img src="./example/replace-anything/man/with_points.png" width="100%"></td>
      <td><img src="./example/replace-anything/man/with_mask.png" width="100%"></td>
      <td><img src="./example/replace-anything/man/replaced_with_mask.png" width="100%"></td>
    </tr>
</table> -->
<p align="center">Text prompt: "a man in office"</p>
<p align="center">
    <img src="./example/GIF/Replace-man.gif" alt="image" style="width:400px;">
</p>

**Click** on an object, **type** in what background you want to replace, and Inpaint Anything will **replace** it!
- Click on an object;
- [SAM](https://segment-anything.com/) segments the object out;
- Input a text prompt;
- Text-prompt-guided inpainting models (e.g., [Stable Diffusion](https://github.com/CompVis/stable-diffusion)) replace the background according to the text.

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
```

### Usage
Download the model checkpoints provided in [segment_anything](./segment_anything/README.md)
(e.g. [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)).

Specify an image, a point and text prompt, and run:
```bash
python fill_anything.py \
    --input_img ./example/replace-anything/dog.png \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt sam_vit_h_4b8939.pth
```

### Demo
<table>
  <caption align="center">Text prompt: "sit on the swing"</caption>
    <tr>
      <td><img src="./example/replace-anything/dog/with_points.png" width="100%"></td>
      <td><img src="./example/replace-anything/dog/with_mask.png" width="100%"></td>
      <td><img src="./example/replace-anything/dog/replaced_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "a bus, on the center of a country road, summer"</caption>
    <tr>
      <td><img src="./example/replace-anything/bus/with_points.png" width="100%"></td>
      <td><img src="./example/replace-anything/bus/with_mask.png" width="100%"></td>
      <td><img src="./example/replace-anything/bus/replaced_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "breakfast"</caption>
    <tr>
      <td><img src="./example/replace-anything/000000029675/with_points.png" width="100%"></td>
      <td><img src="./example/replace-anything/000000029675/with_mask.png" width="100%"></td>
      <td><img src="./example/replace-anything/000000029675/replaced_with_mask.png" width="100%"></td>
    </tr>
</table>

<table>
  <caption align="center">Text prompt: "crossroad in the city"</caption>
    <tr>
      <td><img src="./example/replace-anything/000000000724/with_points.png" width="100%"></td>
      <td><img src="./example/replace-anything/000000000724/with_mask.png" width="100%"></td>
      <td><img src="./example/replace-anything/000000000724/replaced_with_mask.png" width="100%"></td>
    </tr>
</table>

<!-- ## Cite Us -->


## Acknowledgments
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [LaMa](https://github.com/advimman/lama)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)



 ## Other Interesting Repositories
- [Awesome Anything](https://github.com/VainF/Awesome-Anything)
- [Composable AI](https://github.com/Adamdad/Awesome-ComposableAI)
- [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## 📜 Citation
If you find this work useful for your research, please cite us:
```bibtex
@misc{yu2023inpaint,
      title={Inpaint Anything: Segment Anything Meets Image Inpainting}, 
      author={Tao Yu and Runseng Feng and Ruoyu Feng and Jinming Liu and Xin Jin and Wenjun Zeng and Zhibo Chen},
      year={2023},
      eprint={2304.06790},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ## Citation
If you find this project helpful, please cite the following BibTeX entry.
```BibTex
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting}, 
  author={Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng and Zhibo Chen},
  year={2023}
} -->
