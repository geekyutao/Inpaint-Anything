import os
import sys
import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


def load_img_to_array(img_p):
    return np.array(Image.open(img_p))


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str="./lama/configs/prediction/default.yaml",
        mod = 8
):
    assert len(mask.shape) == 2
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(predict_config.device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


if __name__ == '__main__':
    names = ['baseball', 'boat', 'bridge', 'cat',
                        'dog', 'groceries', 'hippopotamus', 'person', 'person_kite', 'person_umbrella']

    for name in names:
        for idx in range(3):
            img_path = f'./example/{name}.jpg'
            mask_path = f'./example/{name}_mask_{idx}.png'
            output_path = f'./example/{name}_inpainted_{idx}.png'
            config_path = './lama/configs/prediction/default.yaml'
            ckpt_path = "/data1/yutao/projects/IAM/lama/big-lama"
            img = load_img_to_array(img_path)
            mask = load_img_to_array(mask_path)
            cur_res = inpaint_img_with_lama(img, mask, config_path, ckpt_path)
            cur_res = Image.fromarray(cur_res)
            cur_res.save(output_path)
