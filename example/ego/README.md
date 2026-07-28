# Egocentric test clip (EgoMimic)

40 frames pulled from [gatech/EgoMimic](https://huggingface.co/datasets/gatech/EgoMimic)
`bowlplace_human.hdf5`, episode `demo_100`. Aria glasses, first-person view of a
hand placing a bowl on a table.

```
frames/                 original RGB frames (front_img_1)
egomimic_masked/        EgoMimic's own hand-masked variant (front_img_1_masked)
comparison/             original | EgoMimic's black-out | our inpainting
```

`egomimic_masked/` is useful twice over: it shows what the dataset authors ship
(the arm painted solid black), and the black pixels recover their exact arm
annotation, which makes it a ready-made mask for testing the inpainting stage
independently of segmentation.

Fetched with HTTP range requests rather than downloading the 40GB archive:

```python
import fsspec, h5py
f = fsspec.open("https://huggingface.co/datasets/gatech/EgoMimic/resolve/main/bowlplace_human.hdf5",
                block_size=8 * 1024 * 1024).open()
h5 = h5py.File(f, "r")
frames = h5["data"]["demo_100"]["obs"]["front_img_1"][:40]
```
