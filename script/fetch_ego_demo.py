"""Pull a few episodes out of a remote egocentric HDF5 dataset.

Uses HTTP range requests, so grabbing a 40-frame clip costs seconds rather than
downloading the whole multi-gigabyte archive.

Example:
    python script/fetch_ego_demo.py --demos demo_100 demo_101 --frames 60
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "egomimic-bowlplace": {
        "url": "https://huggingface.co/datasets/gatech/EgoMimic/resolve/main/bowlplace_human.hdf5",
        "rgb": "front_img_1",
        "masked": "front_img_1_masked",
    },
    "egomimic-groceries": {
        "url": "https://huggingface.co/datasets/gatech/EgoMimic/resolve/main/groceries_human.hdf5",
        "rgb": "front_img_1",
        "masked": "front_img_1_masked",
    },
    "egomimic-clothfold": {
        "url": "https://huggingface.co/datasets/gatech/EgoMimic/resolve/main/smallclothfold_human.hdf5",
        "rgb": "front_img_1",
        "masked": "front_img_1_masked",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="egomimic-bowlplace",
                        choices=list(DATASETS))
    parser.add_argument("--demos", nargs="+", default=["demo_100"],
                        help="Episode keys, or 'list' to just print what exists.")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output_dir", default=str(ROOT / "example" / "ego"))
    parser.add_argument("--with_masked", action="store_true",
                        help="Also pull the dataset's own hand-masked variant, "
                             "whose black pixels double as a free arm mask.")
    args = parser.parse_args()

    import fsspec
    import h5py
    import numpy as np
    from PIL import Image

    cfg = DATASETS[args.dataset]
    print(f"opening {cfg['url']}")
    fh = fsspec.open(cfg["url"], block_size=8 * 1024 * 1024).open()
    h5 = h5py.File(fh, "r")
    data = h5["data"]

    if args.demos == ["list"]:
        keys = list(data.keys())
        print(f"{len(keys)} episodes, first 20: {keys[:20]}")
        return

    out_root = Path(args.output_dir)
    for demo in args.demos:
        if demo not in data:
            print(f"  {demo}: not found, skipping")
            continue
        obs = data[demo]["obs"]
        n = min(args.frames, obs[cfg["rgb"]].shape[0])
        print(f"{demo}: {n} frames", flush=True)

        dst = out_root / demo / "frames"
        dst.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(obs[cfg["rgb"]][:n]):
            Image.fromarray(img).save(dst / f"{i:05d}.jpg", quality=95)
        print(f"  -> {dst}")

        if args.with_masked and cfg["masked"] in obs:
            mdst = out_root / demo / "egomimic_masked"
            mdst.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(obs[cfg["masked"]][:n]):
                Image.fromarray(img).save(mdst / f"{i:05d}.png")
            print(f"  -> {mdst}")


if __name__ == "__main__":
    main()
