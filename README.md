# WAU-Net

PyTorch implementation of **WAU-Net (Wavelet-ASPP U-Net)** for ultrasound image segmentation.

WAU-Net introduces an **ASPP with Wavelet Module (AWWM)** at the U-Net bottleneck. A single-level 2D Haar DWT (`J=1`, `mode='zero'`) decomposes the bottleneck feature map into `LL`, `LH`, `HL`, and `HH`. The three high-frequency subbands are concatenated and processed by ASPP branches with dilation rates **1, 3, and 5**. The processed low-frequency feature, ASPP-enhanced high-frequency features, and original high-frequency subbands are then fused by a projection block.

## Paper

> **Wavelet-ASPP U-Net: Enhancing Ultrasound Image Segmentation with Multi-Scale Context and Frequency-Aware Features**  
> Xin Cheng, Wenbo Yue, Xiaming Wu, Jiahui Xie, Junjie Zhang, Chang Li, Yajun Yu, Xinglong Wu, and Guoping Xu

## Repository Structure

```text
WAU-Net/
├── builders/
│   ├── dataset_builder.py
│   └── model_builder.py
├── datasets/
│   ├── BUS-BRA/{train,val,test,trainval}_list.txt
│   ├── TN3K/{train,val,test,trainval}_list.txt
│   ├── DDTI/{train,val,test,trainval}_list.txt
│   └── BUS_UC_test_list.txt
├── model/
│   ├── WAU-Net.py
│   └── ...
├── utils/
├── preprocess.py
├── train.py
├── test.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md
```

`datasets/` contains the **fixed split files used by the reported experiments**. The image datasets themselves are not redistributed.

## Installation

```bash
git clone https://github.com/apple1986/WAU-Net.git
cd WAU-Net
conda create -n waunet python=3.10 -y
conda activate waunet
pip install -r requirements.txt
```

Verified experimental environment:

```text
Python 3.10.18
PyTorch 2.5.1
Ubuntu 20.04.2 LTS
NVIDIA GeForce RTX 4090 24 GB
pytorch-wavelets 1.3.0
```

## Datasets

The main experiments use three public ultrasound segmentation datasets:

- **BUS-BRA**
- **TN3K**
- **DDTI**

**BUS_UC** is used only as the external target test set for the initial cross-dataset evaluation.

Place the downloaded raw data under `data/` (or pass `--data_root`):

```text
data/
├── BUS-BRA/
│   ├── train/
│   ├── trainannot/
│   ├── val/
│   ├── valannot/
│   ├── test/
│   └── testannot/
├── TN3K/
├── DDTI/
└── BUS_UC/
    ├── test/
    └── testannot/
```

The split-list entries are relative to each dataset root.

### Fixed dataset partitions

| Dataset | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| BUS-BRA | 1875 | 1312 | 188 | 375 |
| TN3K | 3493 | 2445 | 349 | 699 |
| DDTI | 637 | 446 | 64 | 127 |

- BUS-BRA and DDTI were shuffled before the fixed 7:1:2 split.
- TN3K was split according to the original file order.
- The same fixed partitions were used for WAU-Net and compared models.
- The exact split files used in the reported experiments are provided directly in this repository; no new random split should be generated for reproduction.

Complete patient identifiers were not consistently available for all public datasets, so strict patient-level splitting could not be guaranteed for every dataset. This limitation is stated in the manuscript.

## Preprocessing

`preprocess.py` implements the grayscale ultrasound pipeline used by the training/testing loaders:

- single-channel grayscale images
- binary lesion/background masks
- 256 x 256 model input
- random scaling and random horizontal mirroring during training
- training-set mean/std and class-weight computation

## Training configuration

Reported settings:

```text
optimizer: Adam
initial learning rate: 5e-4
weight decay: 1e-4
batch size: 8
maximum epochs: 1000
learning-rate schedule: warm-up polynomial
loss: weighted cross-entropy
random seed: 1234
input size: 256 x 256
```

Additional training configuration material:

[Training Configurations](https://drive.google.com/file/d/10ylye5tKf9Oox3Z63lYL6bKfmmVZlrvc/view?usp=sharing)

## Training

The reported experiments train on the fixed **training split** and keep the validation split separate for validation/model selection.

```bash
python train.py --model WAU-Net --dataset DDTI --gpus 0
python train.py --model WAU-Net --dataset BUS-BRA --gpus 0
python train.py --model WAU-Net --dataset TN3K --gpus 0
```

Use `--data_root /path/to/dataset` when the dataset is not stored under `./data/<dataset>`.

The validation split remains separate from training for checkpoint/model selection. The repository additionally keeps `model_best.pth` using validation mIoU as a practical default, and saves periodic checkpoints every 50 epochs and at the final epoch.

## Model checkpoint

A trained WAU-Net checkpoint for DDTI is available here:

[WAU-Net DDTI Checkpoint](https://drive.google.com/file/d/1vEfBiIQgjf_dtZb8xrxxVvhwlniGbVDm/view?usp=sharing)

## Testing / prediction

```bash
python test.py   --model WAU-Net   --dataset DDTI   --checkpoint ./checkpoint/DDTI/WAU-Net.../model_best.pth   --save --save_prob
```

Binary prediction masks are written to:

```text
result/<dataset>/<model>/
```

Foreground probability maps used for AUC can additionally be written to:

```text
result/<dataset>/<model>/probability/
```

### Cross-dataset evaluation

For the BUS-BRA -> BUS_UC experiment, train/select the model only on BUS-BRA and directly test the resulting BUS-BRA checkpoint on the fixed 162-image BUS_UC test subset without retraining or fine-tuning:

```bash
python test.py   --model WAU-Net   --dataset BUS_UC   --checkpoint /path/to/BUS-BRA/model_best.pth   --save --save_prob
```

When `BUS_UC` is selected, the loader uses the BUS-BRA training-set preprocessing statistics. No BUS_UC training or validation data are used for optimization or model selection.

This experiment is an **initial assessment under dataset shift** and should not be interpreted as conclusive evidence of broad cross-dataset generalization.

## Evaluation

`evaluate_metrics.py` provides:

- Dice
- IoU
- Accuracy
- Sensitivity
- Specificity
- AUC
- HD95 (pixels)

Example:

```bash
python evaluate_metrics.py   --gt-dir ./data/DDTI/testannot   --pred-dir ./result/DDTI/WAU-Net   --prob-dir ./result/DDTI/WAU-Net/probability   --model-name WAU-Net   --out-dir ./eval_results/DDTI
```

For AUC consistent with probability-based evaluation, use probability maps generated by `test.py --save_prob`. If only hard binary masks are supplied, the script can still run but the resulting AUC is only a fallback mask-based approximation.

## Reproducibility

The repository and linked supplementary resources provide:

- WAU-Net model implementation
- training and testing code
- preprocessing code
- fixed data-split files
- training configuration information
- evaluation script
- dependency information
- a trained DDTI checkpoint

The repository deliberately uses the provided fixed partitions rather than regenerating random splits, so the dataset partitions remain identical to those reported in the manuscript.

## Main reported results

| Dataset | Dice | IoU | Accuracy | HD95 (pixels) |
|---|---:|---:|---:|---:|
| BUS-BRA | 0.8872 | 0.8098 | 0.9768 | 12.4526 |
| TN3K | 0.7889 | 0.6905 | 0.9637 | 32.1519 |
| DDTI | 0.7590 | 0.6495 | 0.9700 | 27.5353 |

## Notes on included baseline models

`builders/model_builder.py` registers only model implementations that are actually present in this repository. The paper contains a broader comparison table; models not included here are not exposed as runnable options by the repository.
