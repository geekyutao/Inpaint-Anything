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

TL; DR: Pick any object by clicking it — or just by naming it. With powerful vision models, e.g., [SAM 3](https://github.com/facebookresearch/sam3), [LaMa](https://arxiv.org/abs/2109.07161), [Stable Diffusion](https://arxiv.org/abs/2112.10752) and [ProPainter](https://github.com/sczhou/ProPainter), **Inpaint Anything** removes the object smoothly (i.e., *Remove Anything*). Further, prompted by user input text, Inpaint Anything can fill the object with any desired content (i.e., *Fill Anything*) or replace the background of it arbitrarily (i.e., *Replace Anything*). The same works across **images**, **videos** and **3D scenes**.

> This fork runs on a modernized model stack — SAM 3 for segmentation and video
> tracking, ProPainter for video inpainting, SDXL / FLUX.1-Fill for text-guided
> editing. See [Model stack](#model-stack) for what is used where and how to
> switch. The original SAM 1 / STTN / OSTrack backends all remain available.

> 🤖 **Doing robotics?** [`remove_hands.py`](#robotics) batch-erases human hands
> from egocentric datasets and exports the masks — the *hand removal and
> inpainting* stage of Human-to-Robot synthesis pipelines. On EgoMimic footage it
> matches human annotation at **IoU 0.96** and reconstructs the background
> instead of blacking it out. Jump to
> [Egocentric → Robot data engine](#robotics).

## 📜 News
[2026/7/28] <span style="color:red">🔥NEW</span> **Robotics support: Egocentric → Robot data engine.** `remove_hands.py` batch-erases human hands from egocentric datasets and exports the masks, covering the *hand removal and inpainting* stage of Human-to-Robot synthesis pipelines. Matches human annotation at IoU 0.96 on EgoMimic. See [Egocentric → Robot data engine](#robotics).\
[2026/7/28] <span style="color:red">🔥NEW</span> **Upgraded to [SAM 3](https://github.com/facebookresearch/sam3).** Objects can now be selected by typing a phrase instead of clicking. Video and 3D track with SAM 3's own video predictor, so **OSTrack is no longer needed**. Video inpainting moved to [ProPainter](https://github.com/sczhou/ProPainter) and text-guided editing to SDXL, with [FLUX.1-Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) available. See [Model stack](#model-stack); all legacy backends remain selectable.\
[2023/9/15] [Remove Anything 3D](#remove-anything-3d) code is available!\
[2023/4/30] [Remove Anything Video](#remove-anything-video) available! You can remove any object from a video!\
[2023/4/24] [Local web UI](./app) supported! You can run the demo website locally!\
[2023/4/22] [Website](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything) available! You can experience Inpaint Anything through the interface!\
[2023/4/22] [Remove Anything 3D](#remove-anything-3d) available! You can remove any 3D object from a 3D scene!\
[2023/4/13] [Technical report on arXiv](https://arxiv.org/abs/2304.06790) available!

## 🌟 Features
- [x] 🤖 [**Egocentric → Robot** data engine](#robotics) — batch hand removal for Human-to-Robot pipelines (<span style="color:red">🔥NEW</span>)
- [x] [**Remove** Anything](#remove-anything)
- [x] [**Fill** Anything](#fill-anything)
- [x] [**Replace** Anything](#replace-anything)
- [x] [Remove Anything **3D**](#remove-anything-3d)
- [ ] Fill Anything **3D**
- [ ] Replace Anything **3D**
- [x] [Remove Anything **Video**](#remove-anything-video)
- [ ] Fill Anything **Video**
- [ ] Replace Anything **Video**


## 💡 Highlights
- [x] Select an object by **typing its name** — no clicking needed (<span style="color:red">🔥NEW</span>)
- [x] Video and 3D need **no tracker checkpoint** anymore (<span style="color:red">🔥NEW</span>)
- [x] Swappable backends at every stage, see [Model stack](#model-stack)
- [x] [Local web UI](./app) for both images and video
- [x] Any aspect ratio supported
- [x] 2K resolution supported
- [x] Multiple modalities (i.e., image, video and 3D scene) supported
- [x] [Technical report on arXiv](https://arxiv.org/abs/2304.06790) available
- [x] [Website](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything) available

<!-- ## Updates
| Date | News |
| ------ | --------
| 2023-04-12 | Release the Fill Anything feature | 
| 2023-04-10 | Release the Remove Anything feature |
| 2023-04-10 | Release the first version of Inpaint Anything | -->

## ⚡ Quick start

Requires Python ≥ 3.12, PyTorch ≥ 2.7 and CUDA ≥ 12.6 (SAM 3's floor).

```bash
# 1. dependencies
git clone https://github.com/facebookresearch/sam3.git sam3_repo
python -m pip install -e ./sam3_repo
git clone https://github.com/sczhou/ProPainter.git propainter
python -m pip install -r requirements.txt

# 2. weights — SAM 3 asks you to accept its licence on HuggingFace first, see below
python script/download_weights.py

# 3. run something
bash script/remove_anything.sh          # remove an object from an image
bash script/remove_anything_video.sh    # remove an object from a video
cd app && python app.py                 # local web UI on http://localhost:7860
```

No GPU big enough for SAM 3, or don't want to wait for the approval? Everything
works with the bundled MobileSAM too:
`--sam_model_type vit_t --sam_ckpt ./weights/mobile_sam.pt`.

## <span id="model-stack">🧱 Model stack</span>

| Stage | Default | Alternatives | Selected with |
| --- | --- | --- | --- |
| Segmentation | **SAM 3** | SAM 1 (`vit_h/l/b`), MobileSAM (`vit_t`) | `--sam_model_type` |
| Video / multi-view tracking | **SAM 3 video predictor** | OSTrack *(legacy)* | `--tracker` |
| Object removal (image) | **LaMa** | FLUX.1-Fill, SDXL | `--inpaint_model` |
| Text-guided fill / replace | **SDXL Inpainting** | FLUX.1-Fill, SD 1.5 | `--sd_model` |
| Video inpainting | **ProPainter** | STTN *(legacy)*, per-frame LaMa | `--vi_model` |
| Novel view synthesis (3D) | NeRF | — | — |

Defaults favour what runs comfortably on a single 24GB GPU. LaMa stays the
default for plain object removal because it is far faster than a diffusion model
and still competitive at that specific job; switch to `--inpaint_model flux` when
you want maximum quality and can spare ~20GB of VRAM.

**OSTrack is no longer needed.** Both the video and 3D pipelines track with SAM 3
by default, which removes a model, a manual Google Drive download, and the lossy
mask → box → mask round-trip the old path went through. It stays selectable via
`--tracker ostrack` for reproducing older results.

FLUX.1-Fill is gated on HuggingFace; if you cannot accept the licence there,
`python script/download_weights.py --only flux` pulls the
[ModelScope mirror](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Fill-dev)
into `pretrained_models/FLUX.1-Fill-dev` (~24GB), which you then pass as
`--sd_model ./pretrained_models/FLUX.1-Fill-dev`.

> Licensing note: ProPainter is released under the **NTU S-Lab License 1.0
> (non-commercial)** and FLUX.1-Fill-dev under the **FLUX.1-dev Non-Commercial
> License**. LaMa, SDXL and SAM 3 have their own terms — check them before any
> commercial use.

## <span id="segmentation-backend">🧩 Setup and segmentation backend</span>

All pipelines segment with [SAM 3](https://github.com/facebookresearch/sam3) by
default. Beyond being a stronger segmenter, SAM 3 adds **open-vocabulary text
prompts**: instead of hunting for pixel coordinates, you can pass
`--text_select "dog"` and it will find every matching instance.

### Requirements

SAM 3 needs **Python ≥ 3.12, PyTorch ≥ 2.7 and CUDA ≥ 12.6**. If you are on an
older stack, keep using the SAM 1 backend (see [below](#using-sam-1--mobilesam)).

### Install

```bash
# 1. SAM 3 itself. The folder must NOT be named `sam3`, otherwise it shadows
#    the installed package when scripts run from the repo root.
git clone https://github.com/facebookresearch/sam3.git sam3_repo
python -m pip install -e ./sam3_repo

# 2. ProPainter, used as a source checkout rather than a package
git clone https://github.com/sczhou/ProPainter.git propainter

# 3. Everything else (LaMa, Stable Diffusion, video/3D, web UI)
python -m pip install -r requirements.txt

# 4. Optional: keep the SAM 1 / MobileSAM backend available
python -m pip install -e segment_anything
```

### Download the weights

Everything the default pipelines need lives under `./pretrained_models`. The
helper script fetches it:

```bash
python script/download_weights.py
```

**SAM 3 is a gated HuggingFace repo.** Meta asks you to accept the SAM licence
before downloading, so please get the weights from the official source:

1. Open <https://huggingface.co/facebook/sam3> and click *Agree and access repository*.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Log in locally: `hf auth login` (or `export HF_TOKEN=hf_xxx`).
4. Re-run `python script/download_weights.py --only sam3`.

If HuggingFace is unreachable from your network, the script falls back to the
[ModelScope mirror](https://www.modelscope.cn/models/facebook/sam3) of the same
files. The licence still applies either way — read it at
<https://huggingface.co/facebook/sam3/blob/main/LICENSE>.

> **Use `sam3.pt`, not SAM 3.1.** SAM 3.1's improvements are in multi-object
> video tracking speed (~7x at 128 objects) and need
> `build_sam3_multiplex_video_predictor`; its `sam3.1_multiplex.pt` does not fit
> the builders used here. We measured both on egocentric hand segmentation and
> text-prompt quality is identical (IoU 0.963 vs 0.962 against human
> annotation), but with 3.1 the interactive predictor's weights fail to load, so
> point and box prompts break silently. The code warns if it sees a multiplex
> checkpoint.

The expected layout:

```
pretrained_models/
├── sam3.pt                 # SAM 3, images + video tracking   (gated, see above)
├── config.json             # SAM 3 config, downloaded alongside sam3.pt
├── big-lama/               # LaMa, default image removal      (auto)
│   ├── config.yaml
│   └── models/best.ckpt
├── propainter/             # ProPainter, default video inpainting (auto)
│   ├── ProPainter.pth
│   ├── recurrent_flow_completion.pth
│   └── raft-things.pth
├── FLUX.1-Fill-dev/        # optional, ~24GB   (`--only flux`)
└── sttn.pth                # legacy --vi_model sttn   (`--all`, needs gdown)
pytracking/pretrain/
└── vitb_384_mae_ce_32x4_ep300.pth   # legacy --tracker ostrack only  (manual)
weights/
└── mobile_sam.pt           # MobileSAM, ships with the repo
```

`download_weights.py` fetches SAM 3, LaMa and ProPainter by default — that is
everything the default pipelines need. `--only flux` adds FLUX.1-Fill, `--all`
adds the legacy STTN and OSTrack weights on top. Stable Diffusion weights are
pulled by `diffusers` on first use.

If you would rather not log in, download `sam3.pt` from the model page by hand
and drop it at `./pretrained_models/sam3.pt`; nothing else changes. You can also
point `--sam_ckpt` anywhere else, and if the file is missing the code falls back
to downloading from HuggingFace at runtime.

### Selecting a target with text

Every pipeline accepts `--text_select` as a replacement for `--point_coords`:

```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --text_select "dog" \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --lama_ckpt ./pretrained_models/big-lama
```

A few behavioural notes:

- For images, **all** instances matching the phrase are merged into one mask, so
  "remove the cars" clears every car in one pass. Point prompts still behave as
  before, returning SAM's three candidate granularities.
- For video and 3D, only the highest-scoring instance is kept, because the
  tracker follows a single box.
- Use short noun phrases ("red apple", "a player in white"). Tune recall with
  `--text_confidence` (default `0.5`).
- In `fill_anything.py` / `replace_anything.py`, `--text_select` picks *which
  object* to edit while the existing `--text_prompt` still describes *what to
  generate*.

### <span id="using-sam-1--mobilesam">Using SAM 1 / MobileSAM instead</span>

The old backends still work — pass `--sam_model_type` and a matching checkpoint:

```bash
# MobileSAM: lightest option, checkpoint already in the repo
python remove_anything.py ... --sam_model_type vit_t --sam_ckpt ./weights/mobile_sam.pt

# Original SAM
python remove_anything.py ... --sam_model_type vit_h --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
```

Valid values are `sam3`, `vit_h`, `vit_l`, `vit_b` and `vit_t`. Note that
`--text_select` requires `sam3`.

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


**Click** on an object in the image — or **name** it — and Inpaint Anything will **remove** it instantly!
- Click on an object, or type a phrase like `dog`;
- [SAM 3](https://github.com/facebookresearch/sam3) segments the object out;
- An inpainting model ([LaMa](https://advimman.github.io/lama-project/) by default, or FLUX.1-Fill) fills the "hole".

### Installation
See [Segmentation backend](#segmentation-backend) for the full setup. In short:

```bash
git clone https://github.com/facebookresearch/sam3.git sam3_repo
python -m pip install -e ./sam3_repo
python -m pip install -r requirements.txt
python script/download_weights.py
```

### Usage
Needs `pretrained_models/sam3.pt` and `pretrained_models/big-lama` — see
[Download the weights](#download-the-weights).

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
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```
Or skip the coordinates entirely and name the object:
```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --text_select "dog" \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --lama_ckpt ./pretrained_models/big-lama
```
For harder cases where LaMa leaves smudges, swap in a diffusion model with
`--inpaint_model flux` (FLUX.1-Fill, best quality, ~20GB VRAM) or
`--inpaint_model sdxl`. `--remove_prompt` optionally describes what the
background should look like; leaving it empty just continues the surroundings.

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
- Click on an object, or name it with `--text_select`;
- [SAM 3](https://github.com/facebookresearch/sam3) segments the object out;
- Input a text prompt;
- A text-guided inpainting model ([SDXL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) by default, or FLUX.1-Fill) fills the "hole" according to the text.

### Installation
See [Segmentation backend](#segmentation-backend). Fill Anything needs SAM 3
plus the Stable Diffusion stack, both covered by `requirements.txt`.

### Usage
Needs `pretrained_models/sam3.pt`; the Stable Diffusion weights download
automatically on first run.
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
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt
```
Or name the object to replace instead of clicking it. Here `--text_select`
chooses the target and `--text_prompt` describes what goes in its place:
```bash
python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --text_select "dog" \
    --text_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results
```
`--sd_model` chooses the generator: `sdxl` (default, 1024px), `flux`
(FLUX.1-Fill, best quality, gated on HuggingFace), `sd15`, or any diffusers
inpainting model id. The crop window follows the model's native resolution
automatically.

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
- Click on an object, or name it with `--text_select`;
- [SAM 3](https://github.com/facebookresearch/sam3) segments the object out;
- Input a text prompt;
- A text-guided inpainting model ([SDXL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) by default, or FLUX.1-Fill) replaces the background according to the text.

### Installation
See [Segmentation backend](#segmentation-backend). Same dependencies as Fill
Anything.

### Usage
Needs `pretrained_models/sam3.pt`; the Stable Diffusion weights download
automatically on first run.
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
    --sam_model_type "sam3" \
    --sam_ckpt ./pretrained_models/sam3.pt
```
Or name the foreground to keep. Here `--text_select` chooses what stays and
`--text_prompt` describes the new background:
```bash
python replace_anything.py \
    --input_img ./example/replace-anything/dog.png \
    --text_select "dog" \
    --text_prompt "sit on the swing" \
    --output_dir ./results
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

With a single **click** on an object in the *first* view of source views — or its
name — Remove Anything 3D can remove the object from the *whole* scene!
- Point at (or name) an object in the first source view;
- [SAM 3](https://github.com/facebookresearch/sam3) segments it and propagates the mask across the remaining views with its video predictor;
- [LaMa](https://advimman.github.io/lama-project/) inpaints the object out of every source view;
- [NeRF](https://github.com/yenchenlin/nerf-pytorch) synthesizes novel views of the scene without the object.

### Installation
See [Segmentation backend](#segmentation-backend); `requirements.txt` already
covers the extra `jpeg4py` / `lmdb` needed here.

### Usage
Needs `pretrained_models/sam3.pt` and `pretrained_models/big-lama`; tracking
across views uses SAM 3, so no tracker checkpoint is required. You also need an
LLFF scene (e.g.
[horns](https://drive.google.com/drive/folders/1boi3eK8jNC8yv8IJ7lcL5_F1vutL3imc))
under `./example/3d`.

> Remember to point `datadir` in `nerf/configs/<scene>.txt` at your scene
> directory; it ships pointing at `./data/nerf_llff_data/<scene>`.

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
      --sam_model_type "sam3" \
      --sam_ckpt ./pretrained_models/sam3.pt \
      --lama_config ./lama/configs/prediction/default.yaml \
      --lama_ckpt ./pretrained_models/big-lama \
      --config ./nerf/configs/horns.txt \
      --expname horns
```
`--text_select "horns"` names the object in the first view instead of clicking it. `--mask_idx {0,1,2}` picks one of SAM's three first-view granularities; left unset, SAM 3 decides on its own. `--tracker ostrack` restores the legacy tracker, which then also needs `--tracker_ckpt vitb_384_mae_ce_32x4_ep300` (a *config name*, not a path — it resolves to `./pytracking/pretrain/<name>.pth`).

> Remember to point `datadir` in your NeRF config at the scene directory; the
> erased views are written to `<scene>/images_remove_<factor>/removed_with_mask_<dilate>/`,
> which is where `nerf/load_llff.py` reads them from.


## <span id="remove-anything-video">📌 Remove Anything Video</span>
<table>
    <tr>
      <td><img src="./example/remove-anything-video/paragliding/original.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/paragliding/mask.gif" width="100%"></td>
      <td><img src="./example/remove-anything-video/paragliding/removed.gif" width="100%"></td>
    </tr>
</table>

With a single **click** on an object in the *first* video frame — or just its
name — Remove Anything Video removes it from the *whole* video:
- Point at (or name) an object in the first frame;
- [SAM 3](https://github.com/facebookresearch/sam3) segments it and propagates the mask through every frame with its video predictor;
- [ProPainter](https://github.com/sczhou/ProPainter) fills the hole with temporally consistent content.

### Installation
See [Segmentation backend](#segmentation-backend). Make sure the ProPainter
checkout exists (`git clone https://github.com/sczhou/ProPainter.git propainter`).

### Usage
Needs `pretrained_models/sam3.pt` and `pretrained_models/propainter/`. No
tracker checkpoint is required anymore.
```
bash script/remove_anything_video.sh

```

Specify a video and a point:
```bash
python remove_anything_video.py \
    --input_video ./example/video/paragliding/original_video.mp4 \
    --coords_type key_in \
    --point_coords 652 162 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_ckpt ./pretrained_models/sam3.pt
```
Or name the object instead of clicking it:
```bash
python remove_anything_video.py \
    --input_video ./example/video/paragliding/original_video.mp4 \
    --text_select "paraglider" \
    --dilate_kernel_size 15 \
    --output_dir ./results
```

Useful knobs:

- `--vi_model` picks the inpainter: `propainter` (default), `sttn`, or `lama` (per-frame, no temporal consistency).
- `--vi_fp16` halves ProPainter's memory use. As a rough guide it needs ~11GB at 720x480 for 50 frames in fp32, ~7GB in fp16.
- `--tracker ostrack` restores the old box-tracking path; it then needs `--tracker_ckpt vitb_384_mae_ce_32x4_ep300`, which is a *config name*, not a path (resolved to `./pytracking/pretrain/<name>.pth`).
- `--mask_idx {0,1,2}` picks one of SAM's three first-frame granularities. Left unset, SAM 3 decides on its own.
- `--fps` is only a fallback; the real FPS is read from the video's metadata when present.

The legacy stack is still one flag away:
```bash
python remove_anything_video.py \
    --input_video ./example/video/paragliding/original_video.mp4 \
    --coords_type key_in --point_coords 652 162 --point_labels 1 \
    --output_dir ./results \
    --tracker ostrack --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_model sttn --vi_ckpt ./pretrained_models/sttn.pth --mask_idx 2
```

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

## <span id="robotics">🤖 Egocentric → Robot data engine</span>

Human-to-Robot synthesis pipelines turn egocentric human demonstrations into
robot training data. [Qwen-RobotManip](https://github.com/QwenLM/Qwen-RobotManip)
describes the recipe as *action retargeting → hand removal and inpainting →
simulated robot rendering → depth-guided compositing*, and
[EgoEngine](https://egoengine.github.io/) builds robot observation videos the
same way. **This repo implements the erasure stage**: it removes the human hands
(and optionally the manipulated object) from egocentric footage and hands you
clean plates plus the exact masks the compositing stage needs.

To be explicit about scope — retargeting, robot rendering and depth compositing
are **not** part of this repo. What you get is the visual erasure step, which is
where temporal artifacts would otherwise leak straight into your training set.

```bash
python remove_hands.py \
    --input_dir /data/ego4d/clips \
    --output_dir /data/ego4d_handless \
    --save_masks \
    --output_format png
```

### Dataset layouts

`--input_dir` is walked recursively and every episode is picked up, whether it
is a video file (Ego4D-style) or a folder of frames (EPIC-KITCHENS, TACO, Aria
exports). The output mirrors the input tree, and frame *stems* are preserved so
results still join against your poses, annotations and retargeted actions:

```
/data/ego4d/clips/                     /data/ego4d_handless/
├── clip_a.mp4                 ->      ├── clip_a.mp4
├── take_01/                           ├── take_01/
│   └── frame_0000000000.jpg   ->      │   └── frame_0000000000.png
└── nested/take_02/                    ├── nested/take_02/
    └── 00000.jpg              ->      │   └── 00000.png
                                       ├── _masks/…
                                       └── manifest.json
```

### What gets erased

`--mode keep_object` (default) removes `"hands and forearms"` and leaves the
manipulated object in place — the usual choice, since the rendered robot arm
covers that region anyway. `--mode remove_object` also erases the held object,
for pipelines that re-render it in simulation. `--target "..."` overrides the
phrase entirely; it is fed straight to SAM 3, so anything it understands works
(`"the person"`, `"left hand"`, `"hands and the blue mug"`).

### Built for unattended batch runs

- **No clicking.** Targets are chosen by text, so nothing needs a human in the loop.
- **Models load once** and are reused across every episode.
- **`--skip_existing`** resumes an interrupted run.
- **Hands leave the frame constantly**, so the first frame is not assumed to contain them: several frames are probed before an episode is declared a miss, and single-frame tracking dropouts are filled from neighbours (`--mask_gap_fill`).
- **`manifest.json`** records per-episode status, frame count and mask coverage. Coverage is the quality tripwire: near-zero means segmentation missed, implausibly high means it grabbed the whole scene. Episodes below `--min_coverage` are flagged rather than silently written.
- **Pixels outside the mask are preserved bit-exactly** with `--output_format png`. The default mirrors the input format, which for a JPEG dataset means one extra generation of compression loss — fine for a one-off, worth avoiding when the plates feed further compositing.
- **`--offload` and `--chunk_size`** bound GPU memory on long episodes; `--fp16` roughly halves ProPainter's footprint.

Start with `--dry_run` to see which episodes were discovered, and `--limit 5` to
sanity-check quality before committing to a full pass.

### Test clip included

`example/ego/` holds 40 frames from [EgoMimic](https://huggingface.co/datasets/gatech/EgoMimic)
(Aria glasses, hand placing a bowl) together with EgoMimic's own hand-masked
variant. Grab more with `python script/fetch_ego_demo.py --demos list`, which
reads the remote HDF5 over range requests instead of downloading 40GB.

Running our inpainting against EgoMimic's own arm annotation shows the
difference in approach — they black the arm out, we reconstruct the table
behind it:

<p align="center">
  <img src="./example/ego/demo_100/comparison/original_vs_egomimic_vs_ours_f18.png" width="100%">
  <br><em>original · EgoMimic's black-out · ProPainter reconstruction</em>
</p>

A black blob is an out-of-distribution artifact for whatever policy trains on
these frames; a reconstructed table is not.

### Compositing a robot in afterwards

Putting a robot arm where the human one was is a rendering problem, not an
inpainting one: it needs the arm's 3D pose retargeted to the robot's kinematics
and a URDF rendered through a simulator with calibrated camera extrinsics. That
belongs in your simulation stack, not here.

What this repo gives that stage is its two inputs — the clean plate and the mask
of what was removed. With a rendered robot RGBA sequence, compositing is a few
lines:

```python
import numpy as np
from PIL import Image

plate = np.array(Image.open("out/_plates/take_01/000000.png"))   # --save_plates
robot = np.array(Image.open("robot_render/000000.png"))          # RGBA from your renderer
occluder = np.array(Image.open("bowl_mask/000000.png")) > 0      # optional: stays in front

alpha = robot[:, :, 3:4] / 255.0
alpha = alpha * (1 - occluder[:, :, None])                       # let the object occlude
out = plate * (1 - alpha) + robot[:, :, :3] * alpha
Image.fromarray(out.astype(np.uint8)).save("composited.png")
```

The `occluder` term matters more than it looks: without it the gripper paints
over the manipulated object and reads as pressing down on it rather than holding
it.

We did try generating the robot arm directly with a diffusion inpainter, which
would have avoided the renderer entirely. It does not work — asked for a Franka,
a UR5 and a dexterous hand, SDXL produced an orange cylinder, an orange rod and
a black blob, absorbing colour from nearby objects and ignoring kinematics
completely. Worth knowing so you don't spend time on it.

## Acknowledgments
- [SAM 3](https://github.com/facebookresearch/sam3) — segmentation and video tracking
- [ProPainter](https://github.com/sczhou/ProPainter) — video inpainting
- [LaMa](https://github.com/advimman/lama) — image inpainting
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) / [SDXL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) — text-guided inpainting
- [FLUX.1-Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) — optional high-quality inpainting
- [NeRF](https://github.com/yenchenlin/nerf-pytorch) — novel view synthesis
- [Segment Anything](https://github.com/facebookresearch/segment-anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — legacy segmentation backends
- [OSTrack](https://github.com/botaoye/OSTrack) — legacy video tracking
- [STTN](https://github.com/researchmm/STTN) — legacy video inpainting

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
