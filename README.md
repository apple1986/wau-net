# WAU-Net

PyTorch implementation of **WAU-Net (Wavelet-ASPP U-Net)** for ultrasound image segmentation.

WAU-Net introduces an **ASPP with Wavelet Module (AWWM)** at the U-Net bottleneck. The bottleneck feature map is decomposed by discrete wavelet transform into one low-frequency subband and three high-frequency subbands. ASPP is applied to the high-frequency subbands to enhance multi-scale edge, texture, and boundary representations.

Paper:

> Wavelet-ASPP U-Net: Enhancing Ultrasound Image Segmentation with Multi-Scale Context and Frequency-Aware Features  
> Xin Cheng, Wenbo Yue, Xiaming Wu, Jiahui Xie, Junjie Zhang, Chang Li, Yajun Yu, Xinglong Wu, and Guoping Xu  
> Biomedical Signal Processing and Control, 2026.

---

## Repository Structure

```text
WAU-Net/
├── builders/
├── datasets/
├── model/
├── utils/
├── train.py
├── test.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md
```

- `train.py`: train segmentation models.
- `test.py`: load a checkpoint and generate prediction masks.
- `evaluate_metrics.py`: calculate Dice, IoU, Accuracy, Sensitivity, Specificity, AUC, and HD95 from saved prediction masks.
- `model/`: WAU-Net and comparison model implementations.
- `datasets/`: dataset loaders.
- `utils/`: losses, metrics, optimizers, schedulers, and utility functions.

---

## Installation

```bash
git clone https://github.com/apple1986/WAU-Net.git
cd WAU-Net
```

Create a Python environment:

```bash
conda create -n waunet python=3.10 -y
conda activate waunet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The verified environment used:

```text
Python 3.10.18
PyTorch 2.5.1
CUDA 12.4
GPU: NVIDIA GeForce RTX 4090
pytorch-wavelets 1.3.0
```

---

## Datasets

The experiments were conducted on public ultrasound segmentation datasets, including BUS-BRA, TN3K, DDTI, and BUS_UC.

The released code follows a segmentation dataset format with image folders and mask folders. Please prepare the datasets according to the dataset loader used in `datasets/`.

Example structure:

```text
dataset/
└── camvid/
    ├── train/
    ├── trainannot/
    ├── val/
    ├── valannot/
    ├── test/
    └── testannot/
```

For different datasets, use the same folder convention or modify the corresponding dataset loader.

---

## Training

Example:

```bash
python train.py \
  --model <MODEL_NAME> \
  --dataset camvid \
  --gpus 0
```

Replace `<MODEL_NAME>` with the model name registered in `builders/model_builder.py`.

---

## Testing / Prediction

`test.py` loads a trained checkpoint and saves predicted segmentation masks.

Example:

```bash
python test.py \
  --model <MODEL_NAME> \
  --dataset camvid \
  --checkpoint ./checkpoints/model.pth \
  --save \
  --save_seg_dir ./result/ \
  --gpus 0
```

The prediction masks will be saved under:

```text
result/<dataset>/<model>/
```

---

## Evaluation

Use `evaluate_metrics.py` to calculate evaluation metrics from saved prediction masks.

### Evaluate one model

```bash
python evaluate_metrics.py \
  --gt-dir ./dataset/camvid/testannot \
  --pred-dir ./result/camvid/WAU-Net \
  --model-name WAU-Net \
  --out-dir ./result/eval \
  --out-prefix WAU-Net
```

### Evaluate multiple models

```bash
python evaluate_metrics.py \
  --gt-dir ./dataset/camvid/testannot \
  --out-dir ./result/eval \
  --out-prefix comparison \
  --pred-dirs \
  UNet=./result/camvid/UNet \
  "UNet+ASPP=./result/camvid/UNet_ASPP" \
  "UNet+Wavelet=./result/camvid/UNet_Wavelet" \
  "WAU-Net=./result/camvid/WAU-Net"
```

The script outputs:

```text
*_per_image_metrics.csv
*_summary_metrics.csv
```

Metrics include:

```text
Dice
IoU
Accuracy
Sensitivity
Specificity
AUC
HD95
```

Note: if prediction files are binary masks rather than probability maps, AUC is calculated from normalized grayscale mask values and should be interpreted accordingly.

---

## Results

Main results reported in the paper:

| Dataset | Dice | IoU | Accuracy | HD95 |
|---|---:|---:|---:|---:|
| BUS-BRA | 0.8872 | 0.8098 | 0.9768 | 12.4526 |
| TN3K | 0.7889 | 0.6905 | 0.9637 | 32.1519 |
| DDTI | 0.7590 | 0.6495 | 0.9700 | 27.5353 |

---

## Notes on Baseline Models

Some comparison models in this repository are task-adapted PyTorch implementations for single-channel ultrasound segmentation and two-class output. They are not claimed to be official implementations of the corresponding original papers unless explicitly stated.

For fair comparison, all models should be trained using the same preprocessing, input resolution, optimizer, loss function, train/validation/test split, and evaluation protocol.

---

## Citation

```bibtex
@article{cheng2026waunet,
  title   = {Wavelet-ASPP U-Net: Enhancing Ultrasound Image Segmentation with Multi-Scale Context and Frequency-Aware Features},
  author  = {Cheng, Xin and Yue, Wenbo and Wu, Xiaming and Xie, Jiahui and Zhang, Junjie and Li, Chang and Yu, Yajun and Wu, Xinglong and Xu, Guoping},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {To appear}
}
```

The BibTeX entry will be updated after publication.

---

## Acknowledgement

This project uses the open-source `pytorch_wavelets` package for discrete wavelet transform operations.

---

## Contact

For questions, please open an issue or contact:

Guoping Xu  
xugp@wit.edu.cn
# wau-net
