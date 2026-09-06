import os
import shutil
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from builders.dataset_builder import build_dataset_test
from builders.model_builder import build_model
from utils.metric.metric import get_iou
from utils.utils import save_predict


def parse_args():
    parser = ArgumentParser(description='WAU-Net ultrasound segmentation testing')
    parser.add_argument('--model', default='WAU-Net', help='model name')
    parser.add_argument('--dataset', default='DDTI', choices=['BUS-BRA', 'TN3K', 'DDTI', 'BUS_UC'])
    parser.add_argument('--data_root', type=str, default=None, help='raw dataset root; default: ./data/<dataset>')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint to evaluate')
    parser.add_argument('--save_seg_dir', type=str, default='./result/', help='prediction output root')
    parser.add_argument('--save', action='store_true', help='save predicted binary masks')
    parser.add_argument('--save_prob', action='store_true', help='also save foreground probability maps for AUC')
    parser.add_argument('--cpu', action='store_true', help='run on CPU')
    parser.add_argument('--gpus', default='0', type=str, help='GPU ids, e.g. 0 or 0,1')
    parser.add_argument('--clean', action='store_true', help='clean the model output directory first')
    return parser.parse_args()


def test(args, test_loader, model, device):
    model.eval()
    data_list = []

    for i, (input_tensor, label, _, name) in enumerate(test_loader):
        input_tensor = input_tensor.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            probability = F.softmax(logits, dim=1)[:, 1]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f'[{i + 1}/{len(test_loader)}] time: {elapsed:.4f}s')

        pred_class = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        gt_class = label[0].cpu().numpy().astype(np.uint8)
        pred_scaled = pred_class * 255
        prob = probability[0].cpu().numpy().astype(np.float32)

        if args.save or args.save_prob:
            save_predict(
                pred_scaled,
                gt_class,
                name[0],
                args.save_seg_dir,
                probability=prob if args.save_prob else None,
            )

        data_list.append([gt_class.flatten(), pred_class.flatten()])

    mean_iou, per_class_iou = get_iou(data_list, 2)
    return mean_iou, per_class_iou


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type == 'cuda':
        cudnn.benchmark = True

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)
    if args.clean and os.path.isdir(args.save_seg_dir):
        shutil.rmtree(args.save_seg_dir)
    if args.save or args.save_prob:
        os.makedirs(args.save_seg_dir, exist_ok=True)

    model = build_model(args.model, num_classes=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    # Handle checkpoints saved from DataParallel.
    if any(key.startswith('module.') for key in state.keys()) and not isinstance(model, torch.nn.DataParallel):
        state = {key.replace('module.', '', 1): value for key, value in state.items()}
    model.load_state_dict(state)

    _, test_loader = build_dataset_test(
        args.dataset,
        args.num_workers,
        input_size=(256, 256),
        data_root=args.data_root,
    )
    mean_iou, per_class_iou = test(args, test_loader, model, device)
    print(f'Mean IoU: {mean_iou}')
    print(f'Per-class IoU: {per_class_iou}')


if __name__ == '__main__':
    main()
