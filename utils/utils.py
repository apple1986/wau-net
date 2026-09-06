import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for _, module in feature.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            conv_init(module.weight, **kwargs)
        elif isinstance(module, norm_layer):
            module.eps = bn_eps
            module.momentum = bn_momentum
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_predict(output, gt, img_name, save_path, probability=None):
    """Save binary mask (0/255), GT mask, and optional foreground probability map."""
    os.makedirs(save_path, exist_ok=True)
    Image.fromarray(output.astype(np.uint8)).save(
        os.path.join(save_path, img_name + '_grey.png')
    )
    if gt is not None:
        gt_u8 = (gt.astype(np.uint8) * 255) if np.max(gt) <= 1 else gt.astype(np.uint8)
        Image.fromarray(gt_u8).save(os.path.join(save_path, img_name + '_gt.png'))
    if probability is not None:
        prob_dir = os.path.join(save_path, 'probability')
        os.makedirs(prob_dir, exist_ok=True)
        prob_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(prob_u8).save(os.path.join(prob_dir, img_name + '_prob.png'))


def netParams(model):
    total_parameters = 0
    for parameter in model.parameters():
        p = 1
        for dim in parameter.size():
            p *= dim
        total_parameters += p
    return total_parameters
