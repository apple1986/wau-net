import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import camvid_colorize_mask  # 只导入 camvid_colorize_mask

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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

def save_predict(output, gt, img_name, dataset, save_path, output_grey=True, output_color=False, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output.astype(np.uint8))
        output_grey.save(os.path.join(save_path, img_name + '_grey.png'))

    if output_color:
        if dataset == 'camvid':
            output_color = camvid_colorize_mask(output)  # 输入 [0, 255]
        else:
            raise ValueError("Only camvid dataset is supported for color output")
        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)  # 输入 [0, 255]
        else:
            raise ValueError("Only camvid dataset is supported for gt color")
        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))

def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
    return total_paramters