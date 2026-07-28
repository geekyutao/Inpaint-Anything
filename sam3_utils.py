"""Shared SAM 3 helpers used by both the image and video entry points."""
import contextlib
from pathlib import Path
from typing import Optional

import torch

DEFAULT_SAM3_CKPT = "./pretrained_models/sam3.pt"


def resolve_sam3_ckpt(ckpt_p: Optional[str]) -> Optional[str]:
    """Return a local SAM 3 checkpoint path, or None to fall back to HuggingFace.

    Downloading from HuggingFace requires access to the gated `facebook/sam3`
    repo plus a prior `hf auth login`, so a local file is the preferred route.
    """
    if ckpt_p is None or str(ckpt_p).strip() == "":
        ckpt_p = DEFAULT_SAM3_CKPT
    ckpt_p = Path(ckpt_p).expanduser()
    return str(ckpt_p) if ckpt_p.is_file() else None


def sam3_autocast(device):
    """bfloat16 autocast on Ampere+, as recommended by the SAM 3 examples."""
    if str(device).startswith("cuda") and torch.cuda.is_available() \
            and torch.cuda.get_device_properties(0).major >= 8:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def no_autocast():
    """Opt out of any ambient autocast for the duration of a block.

    SAM 3's tracking predictor enters a global bfloat16 autocast in its
    constructor and deliberately never exits it ("keep using for the entire
    model process"). Every other model in this repo would then silently run in
    bfloat16 — which breaks .numpy() outright and quietly costs precision — so
    the non-SAM inference paths turn it back off around themselves.
    """
    if torch.cuda.is_available():
        return torch.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def warn_if_multiplex_ckpt(ckpt_p: Optional[str]) -> None:
    """SAM 3.1's multiplex checkpoint does not fit the SAM 3 builders.

    Its detector weights load fine, so text prompts still work, but the
    interactive predictor's keys are laid out differently and end up randomly
    initialised — point and box prompts then return garbage with no error. SAM
    3.1's gains are in multi-object video tracking speed, which needs
    `build_sam3_multiplex_video_predictor`, so plain SAM 3 is the right
    checkpoint here.
    """
    if not ckpt_p:
        return
    name = Path(ckpt_p).name.lower()
    if "multiplex" in name or "3.1" in name:
        print(
            f"WARNING: {Path(ckpt_p).name} looks like a SAM 3.1 multiplex "
            f"checkpoint. Text prompts will work, but point and box prompts "
            f"will be silently broken (their weights do not load into this "
            f"model). Use sam3.pt instead."
        )


def build_sam3_image_model(ckpt_p: Optional[str], device: str,
                           enable_inst_interactivity: bool = True):
    from sam3.model_builder import build_sam3_image_model as _build

    warn_if_multiplex_ckpt(ckpt_p)
    local_ckpt = resolve_sam3_ckpt(ckpt_p)
    try:
        return _build(
            device=device,
            checkpoint_path=local_ckpt,
            load_from_HF=local_ckpt is None,
            enable_inst_interactivity=enable_inst_interactivity,
        )
    except Exception as e:
        if local_ckpt is None:
            raise RuntimeError(
                f"Failed to obtain SAM 3 weights automatically ({e}).\n"
                f"Download 'sam3.pt' from https://huggingface.co/facebook/sam3 "
                f"(access must be requested first) and place it at "
                f"'{DEFAULT_SAM3_CKPT}', or pass --sam_ckpt <path>."
            ) from e
        raise
