#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.data import pad_tensor_to_modulo, scale_image

LOGGER = logging.getLogger(__name__)

import imageio
import PIL.Image as Image
from skimage.color import rgb2gray
import torch.nn.functional as F

# if self.scale_factor is not None:
#     result['image'] = scale_image(result['image'], self.scale_factor)
#     result['mask'] = scale_image(result['mask'], self.scale_factor)

# if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
#     result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
#     result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)


def load_img(img_path):
    img = imageio.imread(img_path)
    img = torch.Tensor(img).float() / 255.
    return img

def load_mask(mask_path):
    mask = imageio.imread(mask_path)   # mask must be 255 for hole in this InpaintingModel
    if len(mask.shape) == 3:
        mask = rgb2gray(mask)
    mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
    mask = torch.Tensor(mask).float()
    return mask

def tensor_preprocess(img, mask):
    return img, mask

def rgb_preprocess(img_path, mask_path):
    img = load_img(img_path)    # [H, W, 3]
    mask = load_mask(mask_path) # [H, W]
    return img, mask

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint) # path to xxx.ckpt
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        # if not predict_config.indir.endswith('/'):
        #     predict_config.indir += '/'

        with torch.no_grad():
            batch = {}
            batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
            batch['mask'] = mask[None, None]
            unpad_to_size = [batch['image'].shape[2],  batch['image'].shape[3]]
            batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
            batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            # batch['mask'] = (batch['mask'] == 0) * 1
            '''
            batch['image'].size(): [1, 3, H, W] # 0. ~ 1.
            batch['mask'].size(): [1, 1, H, W]  # unmask 0 / mask 1
            batch['unpad_to_size']: doesn't care
            '''
            batch = model(batch)                    
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            # unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]
            '''
            cur_res.shape: (H, W, 3)
            '''
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cur_out_fname = os.path.join(
                output_path, 
                'result.png'
            )
            cv2.imwrite(cur_out_fname, cur_res)


    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)



if __name__ == '__main__':
    img_path = '/data1/yutao/projects/Inpaint-Anything/example/example.png'
    mask_path = '/data1/yutao/projects/Inpaint-Anything/example/example_mask.png'
    output_path = '/data1/yutao/projects/IAM/lama/myoutputs'
    img, mask = rgb_preprocess(img_path, mask_path)
    mod = 8 #  No need change

    main()

