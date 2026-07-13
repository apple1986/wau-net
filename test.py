import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict
import shutil

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="UNetbam", help="model name: (default SQNet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1, help="the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str, default="/mnt/dat1/xustu1/home/Efficient-Segmentation-Networks-master1/checkpoint/unet+bam/model_1000.pth",
                        help="use the file to load the checkpoint for evaluating or testing")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', action='store_true', help="Get the best result among last few checkpoints")
    parser.add_argument('--save', action='store_true', help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--clean', action='store_true', help="Clean the save directory before running")
    args = parser.parse_args()
    return args

def test(args, test_loader, model):
    model.eval()
    total_batches = len(test_loader)
    data_list = []

    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))

        # 转换为 CPU 和 numpy
        output = output.cpu().data[0].numpy()  # (C, H, W)
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)  # (H, W), 预期 [0, 255]

        # 处理模型输出
        output = output.transpose(1, 2, 0)  # (H, W, C)
        output = np.argmax(output, axis=2)  # (H, W), [0, 1]
        output_scaled = (output * 255).astype(np.uint8)  # 缩放到 [0, 255] 用于保存

        # 转换为类别索引 [0, 1] 用于 IoU 计算
        gt_class = (gt / 255).astype(np.uint8)  # 将 [0, 255] 转换为 [0, 1]
        pred_class = output  # 已是 [0, 1]

        # 调试：打印值范围
        print(f"Batch {i+1}: GT unique values: {np.unique(gt)}, Output unique values: {np.unique(output_scaled)}")
        print(f"Batch {i+1}: GT class values: {np.unique(gt_class)}, Pred class values: {np.unique(pred_class)}")

        # 保存图像（只输出灰度）
        if args.save:
            save_predict(output_scaled, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=True, output_color=False, gt_color=False)

        # 计算 IoU，使用类别索引 [0, 1]
        data_list.append([gt_class.flatten(), pred_class.flatten()])

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu

def test_model(args):
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # 清理输出目录（可选）
    if args.save and args.clean:
        if os.path.exists(args.save_seg_dir):
            shutil.rmtree(args.save_seg_dir)
            print(f"Cleaned save directory: {args.save_seg_dir}")
        os.makedirs(args.save_seg_dir)

    # 构建模型
    model = build_model(args.model, num_classes=args.classes)
    if args.cuda:
        model = model.cuda()
        cudnn.benchmark = True

    if args.save and not args.clean:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # 加载测试集
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)
    args.classes = 2  # 强制二分类，适配 DDTI

    if not args.best:
        if args.checkpoint and os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
        else:
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu = test(args, testLoader, model)
        print(f"Mean IoU: {mIOU_val}")
        print(f"Per class IoU: {per_class_iu}")

    # 保存结果到文件
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        # 最佳模型逻辑略
        pass

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()

if __name__ == '__main__':
    args = parse_args()
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)
    if args.dataset == 'camvid':
        args.classes = 2  # 二分类
    else:
        raise NotImplementedError("Only camvid dataset is supported for now")
    test_model(args)