import importlib.util
import os
from importlib.machinery import SourceFileLoader


def _load_hyphenated_unet(file_name, num_classes):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', file_name)
    module_name = 'wau_net_model'
    loader = SourceFileLoader(module_name, model_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f'Failed to load model file: {file_name}')
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.UNet(classes=num_classes)


def build_model(model_name, num_classes=2):
    """Build only model implementations that are present in this repository."""
    if model_name == 'WAU-Net':
        return _load_hyphenated_unet('WAU-Net.py', num_classes)
    if model_name == 'UNet':
        from model.UNet import UNet
        return UNet(classes=num_classes)
    if model_name == 'AttentionUNet':
        from model.AttentionUNet import AttentionUNet
        return AttentionUNet(classes=num_classes)
    if model_name == 'LinkNet':
        from model.LinkNet import LinkNet
        return LinkNet(classes=num_classes)
    if model_name == 'CGNet':
        from model.CGNet import CGNet
        return CGNet(classes=num_classes)
    if model_name == 'EDANet':
        from model.EDANet import EDANet
        return EDANet(classes=num_classes)
    if model_name == 'LEDNet':
        from model.LEDNet import LEDNet
        return LEDNet(classes=num_classes)
    if model_name == 'ContextNet':
        from model.ContextNet import ContextNet
        return ContextNet(classes=num_classes)

    supported = [
        'WAU-Net', 'UNet', 'AttentionUNet', 'LinkNet',
        'CGNet', 'EDANet', 'LEDNet', 'ContextNet'
    ]
    raise NotImplementedError(
        f"Model {model_name!r} is not included in this repository. Supported models: {supported}"
    )
