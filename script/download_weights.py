"""Fetch the pretrained weights Inpaint-Anything needs.

Usage:
    python script/download_weights.py             # everything that is automatable
    python script/download_weights.py --only sam3 # a single entry

SAM 3 lives behind a gated HuggingFace repo, so it needs a one-off approval
plus a local login before this script can pull it. See the printed hint.
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "pretrained_models"

SAM3_HINT = """
SAM 3 is gated on HuggingFace and the ModelScope mirror did not work either.
To use HuggingFace directly:
  1. Open https://huggingface.co/facebook/sam3 and click "Agree and access repository".
  2. Create a token at https://huggingface.co/settings/tokens (read scope is enough).
  3. Log in locally:   hf auth login      (or:  export HF_TOKEN=hf_xxx)
Then re-run this script.
"""

# ModelScope carries an ungated mirror of the same files, which avoids the
# HuggingFace licence click-through entirely.
MODELSCOPE_SAM3 = "https://www.modelscope.cn/models/facebook/sam3/resolve/master/{}"


def _done(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def download_sam3():
    dst = DEST / "sam3.pt"
    if _done(dst):
        print(f"  sam3.pt already present, skipping ({dst})")
        return True

    # HuggingFace first: it is the canonical source, and works without extra
    # steps for anyone who already accepted the licence.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError
    try:
        for fn in ("config.json", "sam3.pt"):
            src = hf_hub_download(repo_id="facebook/sam3", filename=fn)
            shutil.copy(src, DEST / fn)
        print(f"  saved {dst} (from HuggingFace)")
        return True
    except (GatedRepoError, LocalTokenNotFoundError, OSError) as e:
        print(f"  HuggingFace unavailable ({type(e).__name__}), "
              f"trying the ModelScope mirror")

    try:
        from torch.hub import download_url_to_file
        for fn in ("config.json", "sam3.pt"):
            download_url_to_file(MODELSCOPE_SAM3.format(fn),
                                 str(DEST / fn), progress=True)
        print(f"  saved {dst} (from ModelScope)")
        return True
    except Exception as e:
        print(f"  ModelScope failed too: {type(e).__name__}: {e}")
        print(SAM3_HINT)
        return False


def download_big_lama():
    dst = DEST / "big-lama"
    if _done(dst):
        print(f"  big-lama already present, skipping ({dst})")
        return True
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id="smartywu/big-lama", filename="big-lama.zip")
    with zipfile.ZipFile(src) as z:
        z.extractall(DEST)
    print(f"  saved {dst}")
    return True


def download_propainter():
    """ProPainter, the default video inpainter (3 checkpoints)."""
    from torch.hub import download_url_to_file
    base = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
    dst_dir = DEST / "propainter"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("raft-things.pth", "recurrent_flow_completion.pth",
               "ProPainter.pth"):
        dst = dst_dir / fn
        if _done(dst):
            print(f"  {fn} already present, skipping")
            continue
        download_url_to_file(base + fn, str(dst), progress=True)
        print(f"  saved {dst}")
    return True


def download_flux():
    """FLUX.1-Fill-dev, the optional top-quality inpainting backend (~24GB).

    Gated on HuggingFace, mirrored on ModelScope. Pass the resulting directory
    to --sd_model / --inpaint_model.
    """
    dst = DEST / "FLUX.1-Fill-dev"
    if (dst / "model_index.json").exists():
        print(f"  already present, skipping ({dst})")
        return True
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  needs the modelscope package: pip install modelscope")
        print("  (or accept the licence at "
              "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev "
              "and let diffusers pull it directly)")
        return False
    snapshot_download("black-forest-labs/FLUX.1-Fill-dev", local_dir=str(dst))
    print(f"  saved {dst}")
    print(f"  use it with:  --sd_model {dst}")
    return True


def download_sttn():
    """STTN, only needed for the legacy --vi_model sttn path."""
    dst = DEST / "sttn.pth"
    if _done(dst):
        print(f"  sttn.pth already present, skipping ({dst})")
        return True
    try:
        import gdown
    except ImportError:
        print("  gdown not installed; run 'pip install gdown' or download manually:")
        print("    https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view")
        print(f"    -> {dst}")
        return False
    gdown.download(id="1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv", output=str(dst), quiet=False)
    return _done(dst)


def download_ostrack():
    """OSTrack, only needed for --tracker ostrack and the 3D pipeline."""
    dst = ROOT / "pytracking" / "pretrain" / "vitb_384_mae_ce_32x4_ep300.pth"
    if _done(dst):
        print(f"  OSTrack already present, skipping ({dst})")
        return True
    print("  OSTrack must be downloaded manually from Google Drive:")
    print("    https://drive.google.com/drive/folders/1XJ70dYB6muatZ1LPQGEhyvouX-sU_wnu")
    print(f"    -> {dst}")
    return False


# Everything the default pipelines need, plus the legacy backends last.
TARGETS = {
    "sam3": download_sam3,
    "big-lama": download_big_lama,
    "propainter": download_propainter,
    "flux": download_flux,
    "sttn": download_sttn,
    "ostrack": download_ostrack,
}
# FLUX (~24GB) and the legacy backends are opt-in; the defaults are what the
# standard pipelines need.
DEFAULT_TARGETS = ("sam3", "big-lama", "propainter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(TARGETS), nargs="+",
                        help="Download only these entries.")
    parser.add_argument("--all", action="store_true",
                        help="Also fetch FLUX and the legacy STTN/OSTrack weights.")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    (ROOT / "pytracking" / "pretrain").mkdir(parents=True, exist_ok=True)

    names = args.only or (list(TARGETS) if args.all else list(DEFAULT_TARGETS))
    failed = []
    for name in names:
        print(f"[{name}]")
        if not TARGETS[name]():
            failed.append(name)

    print()
    if failed:
        print(f"Needs manual action: {', '.join(failed)}")
        sys.exit(1)
    print("All requested weights are in place.")


if __name__ == "__main__":
    main()
