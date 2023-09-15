<p align="center">
  <img src="./example/IAM.png">
</p>

# Inpaint Anything: Segment Anything Meets Image Inpainting
Inpaint Anything can inpaint anything in **images**, **videos** and **3D scenes**!
- Authors: Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng and Zhibo Chen.
- Institutes: University of Science and Technology of China; Eastern Institute for Advanced Study.
- [[Paper](https://arxiv.org/abs/2304.06790)] [[Website](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything)] [[Hugging Face Homepage](https://huggingface.co/InpaintAI)]
<p align="center">
  <img src="./example/MainFramework.png" width="100%">
</p>

TL; DR: Users can select any object in an image by clicking on it. With powerful vision models, e.g., [SAM](https://arxiv.org/abs/2304.02643), [LaMa](https://arxiv.org/abs/2109.07161) and [Stable Diffusion (SD)](https://arxiv.org/abs/2112.10752), **Inpaint Anything** is able to remove the object smoothly (i.e., *Remove Anything*). Further, prompted by user input text, Inpaint Anything can fill the object with any desired content (i.e., *Fill Anything*) or replace the background of it arbitrarily (i.e., *Replace Anything*).

## 📜 News
[2023/9/15] [Remove Anything 3D](#remove-anything-3d) code is available!\
[2023/4/30] [Remove Anything Video](#remove-anything-video) available! You can remove any object from a video!\
[2023/4/24] [Local web UI](./app) supported! You can run the demo website locally!\
[2023/4/22] [Website](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything) available! You can experience Inpaint Anything through the interface!\
[2023/4/22] [Remove Anything 3D](#remove-anything-3d) available! You can remove any 3D object from a 3D scene!\
[2023/4/13] [Technical report on arXiv](https://arxiv.org/abs/2304.06790) available!

## 🌟 Features
- [x] [**Remove** Anything](#remove-anything)
- [x] [**Fill** Anything](#fill-anything)
- [x] [**Replace** Anything](#replace-anything)
- [x] [Remove Anything **3D**](#remove-anything-3d) (<span style="color:red">🔥NEW</span>)
- [ ] Fill Anything **3D**
- [ ] Replace Anything **3D**
- [x] [Remove Anything **Video**](#remove-anything-video) (<span style="color:red">🔥NEW</span>)
- [ ] Fill Anything **Video**
- [ ] Replace Anything **Video**


## 💡 Highlights
- [x] Any aspect ratio supported
- [x] 2K resolution supported
- [x] [Technical report on arXiv](https://arxiv.org/abs/2304.06790) available (<span style="color:red">🔥NEW</span>)
- [x] [Website](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything) available (<span style="color:red">🔥NEW</span>)
- [x] [Local web UI](./app) available (<span style="color:red">🔥NEW</span>)
- [x] Multiple modalities (i.e., image, video and 3D scene) supported (<span style="color:red">🔥NEW</span>)

<!-- ## Updates
| Date | News |
| ------ | --------
| 2023-04-12 | Release the Fill Anything feature | 
| 2023-04-10 | Release the Remove Anything feature |
| 2023-04-10 | Release the first version of Inpaint Anything | -->

## <span id="remove-anything">📌 Remove Anything</span>


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
In Windows, we recommend you to first install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and 
open `Anaconda Powershell Prompt (miniconda3)` as administrator.
Then pip install [./lama_requirements_windows.txt](lama_requirements_windows.txt) instead of 
[./lama/requirements.txt](lama%2Frequirements.txt).

### Usage
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [LaMa](./lama/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/remove_anything.sh

```
Specify an image and a point, and Remove Anything will remove the object at the point.
```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --coords_type key_in \
    --point_coords 200 450 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```
You can change `--coords_type key_in` to `--coords_type click` if your machine has a display device. If `click` is set, after running the above command, the image will be displayed. (1) Use *left-click* to record the coordinates of the click. It supports modifying points, and only last point coordinates are recorded. (2) Use *right-click* to finish the selection.

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



## <span id="fill-anything">📌 Fill Anything</span>
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
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/fill_anything.sh

```

Specify an image, a point and text prompt, and run:
```bash
python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
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


## <span id="replace-anything">📌 Replace Anything</span>
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
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) (e.g. [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/replace_anything.sh

```

Specify an image, a point and text prompt, and run:
```bash
python replace_anything.py \
    --input_img ./example/replace-anything/dog.png \
    --coords_type key_in \
    --point_coords 750 500 \
    --point_labels 1 \
    --text_prompt "sit on the swing" \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
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

## <span id="remove-anything-3d">📌 Remove Anything 3D</span>
<!-- Remove Anything 3D can remove any object from a 3D scene! We release some results below. (Code and implementation details will be released soon.) -->

<table>
    <tr>
      <td><img src="./example/remove-anything-3d/horns/org.gif" width="100%"></td>
      <td><img src="./example/remove-anything-3d/horns/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-3d/horns/result.gif" width="100%"></td>
    </tr>
</table>

<table>
    <tr>
      <td><img src="./example/remove-anything-3d/room/org.gif" width="100%"></td>
      <td><img src="./example/remove-anything-3d/room/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-3d/room/result.gif" width="100%"></td>
    </tr>
</table>

With a single **click** on an object in the *first* view of source views, Remove Anything 3D can remove the object from the *whole* scene!
- Click on an object in the first view of source views;
- [SAM](https://segment-anything.com/) segments the object out (with three possible masks);
- Select one mask;
- A tracking model such as [OSTrack](https://github.com/botaoye/OSTrack) is ultilized to track the object in these views;
- SAM segments the object out in each source view according to tracking results;
- An inpainting model such as [LaMa](https://advimman.github.io/lama-project/) is ultilized to inpaint the object in each source view.
- A novel view synthesizing model such as [NeRF](https://github.com/yenchenlin/nerf-pytorch) is ultilized to synthesize novel views of the scene without the object.

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install -r lama/requirements.txt
python -m pip install jpeg4py lmdb
```

### Usage
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [LaMa](./lama/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)), and put them into `./pretrained_models`. Further, download [OSTrack](https://github.com/botaoye/OSTrack) pretrained model from [here](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy) (e.g., [vitb_384_mae_ce_32x4_ep300.pth](https://drive.google.com/drive/folders/1XJ70dYB6muatZ1LPQGEhyvouX-sU_wnu)) and put it into `./pytracking/pretrain`. In addition, download [nerf_llff_data] (e.g, [horns](https://drive.google.com/drive/folders/1boi3eK8jNC8yv8IJ7lcL5_F1vutL3imc)), and put them into `./example/3d`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`. Additionally, download [pretrain](https://drive.google.com/drive/folders/1SERTIfS7JYyOOmXWujAva4CDQf-W7fjv?usp=sharing), put the directory into `./pytracking` and get `./pytracking/pretrain`. 

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/remove_anything_3d.sh

```
Specify a 3d scene, a point, scene config and mask index (indicating using which mask result of the first view), and Remove Anything 3D will remove the object from the whole scene.
```bash
python remove_anything_3d.py \
      --input_dir ./example/3d/horns \
      --coords_type key_in \
      --point_coords 830 405 \
      --point_labels 1 \
      --dilate_kernel_size 15 \
      --output_dir ./results \
      --sam_model_type "vit_h" \
      --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
      --lama_config ./lama/configs/prediction/default.yaml \
      --lama_ckpt ./pretrained_models/big-lama \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --mask_idx 1 \
      --config ./nerf/configs/horns.txt \
      --expname horns
```
The `--mask_idx` is usually set to 1, which typically is the most confident mask result of the first frame. If the object is not segmented out well, you can try other masks (0 or 2).


## <span id="remove-anything-video">📌 Remove Anything Video</span>
<table>
    <tr>
      <td><img src="./example/remove-anything-video/paragliding/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/paragliding/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/paragliding/removed.gif" width="100%"></td>
    </tr>
</table>

With a single **click** on an object in the *first* video frame, Remove Anything Video can remove the object from the *whole* video!
- Click on an object in the first frame of a video;
- [SAM](https://segment-anything.com/) segments the object out (with three possible masks);
- Select one mask;
- A tracking model such as [OSTrack](https://github.com/botaoye/OSTrack) is ultilized to track the object in the video;
- SAM segments the object out in each frame according to tracking results;
- A video inpainting model such as [STTN](https://github.com/researchmm/STTN) is ultilized to inpaint the object in each frame.

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install -r lama/requirements.txt
python -m pip install jpeg4py lmdb
```

### Usage
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [STTN](./sttn/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [sttn.pth](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view)), and put them into `./pretrained_models`. Further, download [OSTrack](https://github.com/botaoye/OSTrack) pretrained model from [here](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy) (e.g., [vitb_384_mae_ce_32x4_ep300.pth](https://drive.google.com/drive/folders/1XJ70dYB6muatZ1LPQGEhyvouX-sU_wnu)) and put it into `./pytracking/pretrain`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`. Additionally, download [pretrain](https://drive.google.com/drive/folders/1SERTIfS7JYyOOmXWujAva4CDQf-W7fjv?usp=sharing), put the directory into `./pytracking` and get `./pytracking/pretrain`.

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/remove_anything_video.sh

```

Specify a video, a point, video FPS and mask index (indicating using which mask result of the first frame), and Remove Anything Video will remove the object from the whole video.
```bash
python remove_anything_video.py \
    --input_video ./example/video/paragliding/original_video.mp4 \
    --coords_type key_in \
    --point_coords 652 162 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt ./pretrained_models/sttn.pth \
    --mask_idx 2 \
    --fps 25
```
The `--mask_idx` is usually set to 2, which typically is the most confident mask result of the first frame. If the object is not segmented out well, you can try other masks (0 or 1).

### Demo
<table>
    <tr>
      <td><img src="./example/remove-anything-video/drift-chicane/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/drift-chicane/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/drift-chicane/removed.gif" width="100%"></td>
    </tr>
</table>

<table>
    <tr>
      <td><img src="./example/remove-anything-video/surf/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/surf/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/surf/removed.gif" width="100%"></td>
    </tr>
</table>

<table>
    <tr>
      <td><img src="./example/remove-anything-video/tennis-vest/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/tennis-vest/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/tennis-vest/removed.gif" width="100%"></td>
    </tr>
</table>

<table>
    <tr>
      <td><img src="./example/remove-anything-video/dog-gooses/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/dog-gooses/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/dog-gooses/removed.gif" width="100%"></td>
    </tr>
</table>

## Acknowledgments
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [LaMa](https://github.com/advimman/lama)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [OSTrack](https://github.com/botaoye/OSTrack)
- [STTN](https://github.com/researchmm/STTN)

 ## Other Interesting Repositories
- [Awesome Anything](https://github.com/VainF/Awesome-Anything)
- [Composable AI](https://github.com/Adamdad/Awesome-ComposableAI)
- [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Citation
If you find this work useful for your research, please cite us:
```
@article{yu2023inpaint,
  title={Inpaint Anything: Segment Anything Meets Image Inpainting},
  author={Yu, Tao and Feng, Runseng and Feng, Ruoyu and Liu, Jinming and Jin, Xin and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2304.06790},
  year={2023}
}
```
  
<p align="center">
  <a href="https://star-history.com/#geekyutao/Inpaint-Anything&Date">
    <img src="https://api.star-history.com/svg?repos=geekyutao/Inpaint-Anything&type=Date" alt="Star History Chart">
  </a>
</p>
