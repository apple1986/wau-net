# WAU-Net

PyTorch implementation of **WAU-Net (Wavelet-ASPP U-Net)** for ultrasound image segmentation.

WAU-Net introduces an **ASPP with Wavelet Module (AWWM)** at the U-Net bottleneck. The bottleneck feature map is decomposed by discrete wavelet transform into one low-frequency subband and three high-frequency subbands. ASPP is applied to the high-frequency subbands to enhance multi-scale edge, texture, and boundary representations.

## Paper

> **Wavelet-ASPP U-Net: Enhancing Ultrasound Image Segmentation with Multi-Scale Context and Frequency-Aware Features**  
> Xin Cheng, Wenbo Yue, Xiaming Wu, Jiahui Xie, Junjie Zhang, Chang Li, Yajun Yu, Xinglong Wu, and Guoping Xu

---

## Repository Structure

```text
WAU-Net/
├── builders/
├── datasets/
├── model/
├── utils/
├── configs/
├── preprocess.py
├── train.py
├── test.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md
```

- `preprocess.py`: preprocessing and data preparation utilities.
- `train.py`: train segmentation models.
- `test.py`: load a checkpoint and generate prediction masks.
- `evaluate_metrics.py`: calculate Dice, IoU, Accuracy, Sensitivity, Specificity, AUC, and HD95 from saved prediction masks.
- `configs/`: dataset-specific training configuration files.
- `model/`: WAU-Net and comparison model implementations.
- `datasets/`: dataset loaders and dataset organization.
- `utils/`: losses, metrics, optimizers, schedulers, and utility functions.

---

## Installation

Clone the repository:

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

The verified environment used in the experiments was:

```text
Python 3.10.18
PyTorch 2.5.1
CUDA 12.4
GPU: NVIDIA GeForce RTX 4090
pytorch-wavelets 1.3.0
```

---

## Datasets

The main experiments were conducted on three public ultrasound segmentation datasets:

- **BUS-BRA**
- **TN3K**
- **DDTI**

An additional external cross-dataset experiment was conducted by training on **BUS-BRA** and testing on **BUS_UC**.

The datasets themselves are not redistributed in this repository. Please obtain them from their original public sources and follow the corresponding licenses and terms of use.

### Dataset Partitioning

The datasets were **split before model training**, and the same fixed training, validation, and testing partitions were used for WAU-Net and all baseline models to ensure a fair comparison.

A custom **7:1:2** split protocol was used for the three main datasets:

| Dataset | Total images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| BUS-BRA | 1875 | 1312 | 188 | 375 |
| TN3K | 3493 | 2445 | 349 | 699 |
| DDTI | 637 | 446 | 64 | 127 |

The partitioning procedure was:

- **BUS-BRA:** the dataset was shuffled before applying the 7:1:2 split.
- **DDTI:** the dataset was shuffled before applying the 7:1:2 split.
- **TN3K:** the original file order was used to construct the 7:1:2 split.

The partitions were fixed after splitting and were reused for every compared model. Therefore, users who wish to reproduce the reported experiments should use the same prepared dataset partitions rather than creating new random splits.

### Note on Patient-Level Separation

Complete patient identifiers were not consistently available for all public datasets. Therefore, strict patient-level separation could not be guaranteed for every dataset. This limitation is acknowledged in the manuscript, and future validation will use strictly patient-level and multi-center data partitions when such information is available.

---

## Preprocessing

Preprocessing is implemented in `preprocess.py` together with the dataset loaders.

The experimental pipeline includes:

- grayscale ultrasound image loading
- binary mask processing
- resizing/cropping to 256 × 256
- random flipping and random scaling for training augmentation
- training-set statistics and class-weight computation where required

All input images used in the experiments were processed at a resolution of **256 × 256**.

---

## Training Configuration

You can download the full training configuration files here:

[Training Configurations](https://drive.google.com/file/d/10ylye5tKf9Oox3Z63lYL6bKfmmVZlrvc/view?usp=sharing)

The configuration files include:
- input size
- batch size
- learning rate
- optimizer
- training epochs
- random seed
- data augmentation settings

---

## Training

Example:

```bash
python train.py   --model <MODEL_NAME>   --dataset xxx_dataset   --gpus 0
```

Replace `<MODEL_NAME>` with the model name registered in `builders/model_builder.py`.

---

## Model Checkpoint

A trained WAU-Net checkpoint for DDTI is available for download:

[WAU-Net DDTI Checkpoint](https://drive.google.com/file/d/1vEfBiIQgjf_dtZb8xrxxVvhwlniGbVDm/view?usp=sharing)

The checkpoint is provided to support reproduction of the inference and evaluation pipeline.

---

## Testing / Prediction

`test.py` loads a trained checkpoint and saves predicted segmentation masks.

Example:

```bash
python test.py   --model <MODEL_NAME>   --dataset xxx_dataset   --checkpoint ./checkpoints/model.pth   --save   --save_seg_dir ./result/   --gpus 0
```

The prediction masks will be saved under:

```text
result/<dataset>/<model>/
```

---

## Evaluation

The evaluation script `evaluate_metrics.py` calculates the segmentation metrics used in the paper:

- Dice coefficient
- IoU
- Accuracy
- Sensitivity
- Specificity
- AUC
- HD95

Basic example:

```bash
python evaluate_metrics.py
```

Example for evaluating multiple models:

```bash
python evaluate_metrics.py   --gt-dir ./dataset/xxx_dataset/testannot   --out-dir ./result/eval   --out-prefix comparison   --pred-dirs   UNet=./result/xxx_dataset/UNet   "UNet+ASPP=./result/xxx_dataset/UNet_ASPP"   "UNet+Wavelet=./result/xxx_dataset/UNet_Wavelet"   "WAU-Net=./result/xxx_dataset/WAU-Net"
```

The script outputs:

```text
*_per_image_metrics.csv
*_summary_metrics.csv
```

**AUC note:** if prediction files are binary masks rather than probability maps, AUC is calculated from normalized grayscale mask values and should be interpreted accordingly.

---

## Reproducibility

The repository provides the main materials needed to reproduce the reported experiments:

- model implementation
- training and testing scripts
- preprocessing instructions/code
- evaluation scripts
- configuration files
- dataset partition information
- dependency information
- a trained WAU-Net checkpoint for DDTI

For fair comparison and reproducibility, **the same fixed dataset partitions were used for WAU-Net and all baseline models**.

---

## Results

Main results reported in the paper:

| Dataset | Dice | IoU | Accuracy | HD95 |
|---|---:|---:|---:|---:|
| BUS-BRA | 0.8872 | 0.8098 | 0.9768 | 12.4526 |
| TN3K | 0.7889 | 0.6905 | 0.9637 | 32.1519 |
| DDTI | 0.7590 | 0.6495 | 0.9700 | 27.5353 |

For the cross-dataset experiment, models were trained on BUS-BRA and tested on BUS_UC. The experiment is intended as an additional assessment under dataset shift and should not be interpreted as conclusive evidence of broad cross-dataset generalization.

---

## Notes on Baseline Models

Some comparison models in this repository are task-adapted PyTorch implementations for single-channel ultrasound segmentation and two-class output. They are not claimed to be official implementations of the corresponding original papers unless explicitly stated.

For fair comparison, all models should be trained using the same preprocessing, input resolution, optimizer, loss function, train/validation/test split, and evaluation protocol.

---

## Citation

The manuscript is currently under review. The citation information will be updated after publication.

```bibtex
@misc{cheng2026waunet,
  title  = {Wavelet-ASPP U-Net: Enhancing Ultrasound Image Segmentation with Multi-Scale Context and Frequency-Aware Features},
  author = {Cheng, Xin and Yue, Wenbo and Wu, Xiaming and Xie, Jiahui and Zhang, Junjie and Li, Chang and Yu, Yajun and Wu, Xinglong and Xu, Guoping},
  year   = {2026},
  note   = {Manuscript submitted to Biomedical Signal Processing and Control}
}
```

---

## Acknowledgement

This project uses the open-source `pytorch_wavelets` package for discrete wavelet transform operations.

---

## Contact

For questions, please open an issue or contact:

Guoping Xu  
xugp@wit.edu.cn
