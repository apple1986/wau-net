import os
import os.path as osp
import pickle
import random

import cv2
import numpy as np
from torch.utils import data


def _subsample_img_ids(img_ids, train_fraction, subset_seed):
    """Keep a deterministic subset of list entries; train_fraction is in (0, 1]."""
    if train_fraction >= 1.0 - 1e-9:
        return img_ids
    n = len(img_ids)
    k = max(1, int(round(n * float(train_fraction))))
    rng = np.random.RandomState(subset_seed)
    pick = rng.permutation(n)[:k]
    return [img_ids[i] for i in sorted(pick)]


def _read_list(list_path):
    with open(list_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(
                f"Each split-list line must contain image and mask paths: {list_path}: {line}"
            )
    return lines


def _to_binary_mask(label):
    # Public datasets in this repository use binary lesion/background masks.
    # This also safely handles masks stored as 0/1 or 0/255.
    return (label > 127).astype(np.uint8) if label.max() > 1 else label.astype(np.uint8)


class UltrasoundDataSet(data.Dataset):
    """Training dataset for BUS-BRA, TN3K and DDTI."""

    def __init__(self, root='', list_path='', max_iters=None, crop_size=(256, 256),
                 mean=128.0, scale=True, mirror=True, ignore_label=255,
                 train_fraction=1.0, subset_seed=1234):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = float(np.asarray(mean).reshape(-1)[0])
        self.is_mirror = mirror

        self.img_ids = _read_list(list_path)
        self.img_ids = _subsample_img_ids(self.img_ids, train_fraction, subset_seed)
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for entry in self.img_ids:
            image_rel, label_rel = entry.split()[:2]
            self.files.append({
                'img': osp.join(self.root, image_rel),
                'label': osp.join(self.root, label_rel),
                'name': osp.splitext(osp.basename(image_rel))[0],
            })
        print(f"Training samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item['img'], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {item['img']}")
        if label is None:
            raise FileNotFoundError(f"Unable to read mask: {item['label']}")

        original_size = image.shape
        label = _to_binary_mask(label)

        if self.scale:
            f_scale = random.choice([0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) - self.mean

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]

        image = image[None, :, :]
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(original_size), item['name']


class UltrasoundValDataSet(data.Dataset):
    """Validation/test dataset with deterministic 256x256 preprocessing."""

    def __init__(self, root='', list_path='', image_size=(256, 256), mean=128.0,
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.image_size = tuple(image_size)
        self.ignore_label = ignore_label
        self.mean = float(np.asarray(mean).reshape(-1)[0])

        self.img_ids = _read_list(list_path)
        self.files = []
        for entry in self.img_ids:
            image_rel, label_rel = entry.split()[:2]
            self.files.append({
                'img': osp.join(self.root, image_rel),
                'label': osp.join(self.root, label_rel),
                'name': osp.splitext(osp.basename(image_rel))[0],
            })
        print(f"Evaluation samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = cv2.imread(item['img'], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {item['img']}")
        if label is None:
            raise FileNotFoundError(f"Unable to read mask: {item['label']}")

        original_size = image.shape
        width, height = self.image_size[1], self.image_size[0]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
        label = _to_binary_mask(label)

        image = image.astype(np.float32) - self.mean
        image = image[None, :, :]
        return image.copy(), label.copy(), np.array(original_size), item['name']


class UltrasoundTrainInform:
    """Compute training-set mean/std and class weights used by weighted CE loss."""

    def __init__(self, data_dir='', classes=2, train_set_file='', inform_data_file='', normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(1, dtype=np.float32)
        self.std = np.zeros(1, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        norm_hist = histogram / np.maximum(np.sum(histogram), 1.0)
        for i in range(self.classes):
            self.classWeights[i] = 1.0 / np.log(self.normVal + norm_hist[i])

    def collectDataAndSave(self):
        entries = _read_list(self.train_set_file)
        global_hist = np.zeros(self.classes, dtype=np.float64)

        for entry in entries:
            image_rel, label_rel = entry.split()[:2]
            image_path = osp.join(self.data_dir, image_rel)
            label_path = osp.join(self.data_dir, label_rel)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")
            if label is None:
                raise FileNotFoundError(f"Unable to read mask: {label_path}")

            label = _to_binary_mask(label)
            hist = np.bincount(label.reshape(-1), minlength=self.classes)[:self.classes]
            global_hist += hist
            self.mean[0] += float(np.mean(image))
            self.std[0] += float(np.std(image))

        count = max(len(entries), 1)
        self.mean /= count
        self.std /= count
        self.compute_class_weights(global_hist)

        data_dict = {
            'mean': self.mean,
            'std': self.std,
            'classWeights': self.classWeights,
        }
        os.makedirs(osp.dirname(self.inform_data_file), exist_ok=True)
        with open(self.inform_data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        return data_dict
