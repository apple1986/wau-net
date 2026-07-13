from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet as BaseUNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
from model.AttentionUNet import AttentionUNet

import importlib.util
import os
from importlib.machinery import SourceFileLoader


def load_custom_unet(module_name, file_name, num_classes):
    custom_model_path = os.path.join(
        os.path.dirname(__file__), '..', 'model', file_name
    )

    loader = SourceFileLoader(module_name, custom_model_path)
    spec = importlib.util.spec_from_loader(module_name, loader)

    if spec is None:
        raise ImportError("Failed to load custom model spec: {}".format(file_name))

    custom_module = importlib.util.module_from_spec(spec)
    loader.exec_module(custom_module)

    return custom_module.UNet(classes=num_classes)


def build_model(model_name, num_classes):
    if model_name == 'SQNet':
        return SQNet(classes=num_classes)

    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)

    elif model_name == 'SegNet':
        return SegNet(classes=num_classes)

    elif model_name == 'UNet':
        return BaseUNet(classes=num_classes)

    elif model_name == 'ENet':
        return ENet(classes=num_classes)

    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)

    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)

    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)

    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)

    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)

    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)

    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)

    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)

    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)

    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)

    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)

    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)

    # ===================== abunet 1~9 =====================
    elif model_name == 'abunet1':
        return load_custom_unet('custom_abunet1', 'abunet1.py', num_classes)
    elif model_name == 'abunet':
        return load_custom_unet('custom_abunet', 'abunet.py', num_classes)

    elif model_name == 'abunet2':
        return load_custom_unet('custom_abunet2', 'abunet2.py', num_classes)

    elif model_name == 'abunet3':
        return load_custom_unet('custom_abunet3', 'abunet3.py', num_classes)

    elif model_name == 'abunet4':
        return load_custom_unet('custom_abunet4', 'abunet4.py', num_classes)

    elif model_name == 'abunet5':
        return load_custom_unet('custom_abunet5', 'abunet5.py', num_classes)

    elif model_name == 'abunet6':
        return load_custom_unet('custom_abunet6', 'abunet6.py', num_classes)

    elif model_name == 'abunet7':
        return load_custom_unet('custom_abunet7', 'abunet7.py', num_classes)

    elif model_name == 'abunet8':
        return load_custom_unet('custom_abunet8', 'abunet8.py', num_classes)

    elif model_name == 'abunet9':
        return load_custom_unet('custom_abunet9', 'abunet9.py', num_classes)

    # ===================== WAUNet 1~9 =====================
    elif model_name == 'WAUNet1':
        from model.WAUNet1 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet2':
        from model.WAUNet2 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet3':
        from model.WAUNet3 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet4':
        from model.WAUNet4 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet5':
        from model.WAUNet5 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet6':
        from model.WAUNet6 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet7':
        from model.WAUNet7 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet8':
        from model.WAUNet8 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    elif model_name == 'WAUNet9':
        from model.WAUNet9 import UNet as WAUNetModel
        return WAUNetModel(classes=num_classes)

    # ===================== custom UNet models =====================
    elif model_name == 'UNetbam':
        return load_custom_unet(
            'custom_unet_bam',
            'UNetbam.py',
            num_classes
        )

    elif model_name == 'unet+bam+jie1aspp':
        return load_custom_unet(
            'custom_unet_bam_jie1aspp',
            'unet+bam+jie1aspp.py',
            num_classes
        )

    elif model_name == 'unet+pingjingaspp':
        return load_custom_unet(
            'custom_unet_pingjingaspp',
            'unet+pingjingaspp.py',
            num_classes
        )

    elif model_name == 'unet+pingjingbo':
        return load_custom_unet(
            'custom_unet_pingjingbo',
            'unet+pingjingbo.py',
            num_classes
        )

    elif model_name == 'unet+pingjingboaspp':
        return load_custom_unet(
            'custom_unet_pingjingboaspp',
            'unet+pingjingboaspp.py',
            num_classes
        )

    elif model_name == 'unet+pingjingboaspp_db2':
        return load_custom_unet(
            'custom_unet_pingjingboaspp_db2',
            'unet+pingjingboaspp_db2.py',
            num_classes
        )

    elif model_name == 'unet+pingjingboaspp_db4':
        return load_custom_unet(
            'custom_unet_pingjingboaspp_db4',
            'unet+pingjingboaspp_db4.py',
            num_classes
        )

    elif model_name == 'unet+pingjingboaspp_sym2':
        return load_custom_unet(
            'custom_unet_pingjingboaspp_sym2',
            'unet+pingjingboaspp_sym2.py',
            num_classes
        )

    elif model_name == 'unet+pingjingboaspp_coif1':
        return load_custom_unet(
            'custom_unet_pingjingboaspp_coif1',
            'unet+pingjingboaspp_coif1.py',
            num_classes
        )

    elif model_name.startswith('unet+pingjingboaspp_j'):
        j_suffix = model_name[len('unet+pingjingboaspp_j'):]

        if j_suffix not in ('1', '2', '3'):
            raise ValueError(
                "Unknown wavelet-depth ablation '{}'; use "
                "unet+pingjingboaspp_j1, unet+pingjingboaspp_j2, "
                "or unet+pingjingboaspp_j3.".format(model_name)
            )

        return load_custom_unet(
            'custom_unet_pingjingboaspp_j{}'.format(j_suffix),
            'unet+pingjingboaspp_j{}.py'.format(j_suffix),
            num_classes
        )

    elif model_name.startswith('unet+pingjingboaspp_dil_'):
        dil_suffixes = ('1', '13', '1357', '123', '147', '246', '369', '61218')
        suffix = model_name[len('unet+pingjingboaspp_dil_'):]

        if suffix not in dil_suffixes:
            raise ValueError(
                "Unknown ASPP dilation ablation '{}'; expected suffix one of: {}".format(
                    model_name,
                    ', '.join('unet+pingjingboaspp_dil_' + s for s in dil_suffixes)
                )
            )

        return load_custom_unet(
            'custom_unet_pingjingboaspp_dil_{}'.format(suffix),
            'unet+pingjingboaspp_dil_{}.py'.format(suffix),
            num_classes
        )

    # ==================== 7 个对比模型 ====================
    elif model_name == 'AttentionUNet':
        return AttentionUNet(in_channels=1, classes=num_classes)

    elif model_name == 'TransUNet':
        from model.TransUNet import TransUNet
        return TransUNet(in_channels=1, classes=num_classes)

    elif model_name == 'LeViT-UNet':
        from model.LeViTUNet import LeViTUNet
        return LeViTUNet(in_channels=1, classes=num_classes)

    elif model_name == 'Swin-UNet':
        from model.SwinUNet import SwinUNet
        return SwinUNet(in_channels=1, classes=num_classes)

    elif model_name == 'SegNeXt':
        from model.SegNeXt import SegNext
        return SegNext(in_channels=1, classes=num_classes)

    elif model_name == 'SegFormer':
        from model.SegFormer import SegFormer
        return SegFormer(in_channels=1, classes=num_classes)

    elif model_name == 'SK-UNet':
        from model.SKUNet import SKUNet
        return SKUNet(in_channels=1, classes=num_classes)

    elif model_name == 'UNetPP':
        from model.UNetPP import UNetPP
        return UNetPP(in_channels=1, classes=num_classes)
        # ==================== Transformer + AWWM models ====================
    elif model_name == 'PVTUNet':
        from model.transformer_awwm_models import PVTUNet
        return PVTUNet(in_channels=1, classes=num_classes)

    elif model_name == 'PVTUNet_AWWM':
        from model.transformer_awwm_models import PVTUNet_AWWM
        return PVTUNet_AWWM(in_channels=1, classes=num_classes)

    elif model_name == 'CvTUNet':
        from model.transformer_awwm_models import CvTUNet
        return CvTUNet(in_channels=1, classes=num_classes)

    elif model_name == 'CvTUNet_AWWM':
        from model.transformer_awwm_models import CvTUNet_AWWM
        return CvTUNet_AWWM(in_channels=1, classes=num_classes)

    elif model_name == 'MobileViTUNet':
        from model.transformer_awwm_models import MobileViTUNet
        return MobileViTUNet(in_channels=1, classes=num_classes)

    elif model_name == 'MobileViTUNet_AWWM':
        from model.transformer_awwm_models import MobileViTUNet_AWWM
        return MobileViTUNet_AWWM(in_channels=1, classes=num_classes)

    # ==================== 新增对比模型 ====================
    elif model_name == 'CMUNet':
        return load_custom_unet(
            'custom_cmunet',
            'CMUNet.py',
            num_classes
        )

    elif model_name == 'DAUSNet':
        return load_custom_unet(
            'custom_dausnet',
            'DAUSNet.py',
            num_classes
        ) 

    elif model_name == 'MaxViTUNet':
        from model.transformer_awwm_models import MaxViTUNet
        return MaxViTUNet(in_channels=1, classes=num_classes)

    elif model_name == 'MaxViTUNet_AWWM':
        from model.transformer_awwm_models import MaxViTUNet_AWWM
        return MaxViTUNet_AWWM(in_channels=1, classes=num_classes)
    elif model_name == 'NUNet':
        return load_custom_unet(
            'custom_nunet',
            'NUNet.py',
            num_classes   ) 

    else:
        raise ValueError(
            "Unsupported model '{}'. Available models: SQNet, LinkNet, SegNet, UNet, ENet, ERFNet, CGNet, "
            "EDANet, ESNet, ESPNet, LEDNet, ESPNet_v2, ContextNet, FastSCNN, DABNet, FSSNet, FPENet, "
            "UNetbam, unet+bam+jie1aspp, unet+pingjingaspp, unet+pingjingbo, unet+pingjingboaspp, "
            "unet+pingjingboaspp_db2, unet+pingjingboaspp_db4, unet+pingjingboaspp_sym2, unet+pingjingboaspp_coif1, "
            "unet+pingjingboaspp_dil_1, unet+pingjingboaspp_dil_13, unet+pingjingboaspp_dil_1357, unet+pingjingboaspp_dil_123, "
            "unet+pingjingboaspp_dil_147, unet+pingjingboaspp_dil_246, unet+pingjingboaspp_dil_369, unet+pingjingboaspp_dil_61218, "
            "unet+pingjingboaspp_j1, unet+pingjingboaspp_j2, unet+pingjingboaspp_j3,NUNet "
            "AttentionUNet, TransUNet, LeViT-UNet, Swin-UNet, SegNeXt, SegFormer, SK-UNet, "
            "WAUNet1, WAUNet2, WAUNet3, WAUNet4, WAUNet5, WAUNet6, WAUNet7, WAUNet8,, WAUNet9, UNetPP, "
            "abunet1,abunet, abunet2, abunet3, abunet4, abunet5, abunet6, abunet7, abunet8, abunet9,CMUNet, DAUSNet,"
            "PVTUNet, PVTUNet_AWWM, CvTUNet, CvTUNet_AWWM, MobileViTUNet, MobileViTUNet_AWWM, MaxViTUNet, MaxViTUNet_AWWM"
            .format(model_name)
        )