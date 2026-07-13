import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import binary_erosion, distance_transform_edt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate binary ultrasound segmentation masks."
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory containing ground-truth masks.")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Directory containing prediction masks for a single model.")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Model name for single-model evaluation.")
    parser.add_argument(
        "--pred-dirs",
        nargs="*",
        default=None,
        help=(
            "Multiple prediction directories in the format Name=Path. "
            "Example: --pred-dirs UNet=./result/UNet WAU-Net=./result/WAU-Net"
        ),
    )
    parser.add_argument("--out-dir", type=str, default="./eval_results",
                        help="Directory for saving CSV results.")
    parser.add_argument("--out-prefix", type=str, default="metrics",
                        help="Prefix of output CSV files.")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Resize masks to this square size before evaluation.")
    parser.add_argument("--threshold", type=int, default=127,
                        help="Threshold for converting grayscale masks to binary masks.")
    return parser.parse_args()


def list_image_files(folder):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    folder = Path(folder)
    files = []
    for ext in exts:
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(files)


def find_prediction_file(pred_dir, gt_path):
    pred_dir = Path(pred_dir)
    stem = gt_path.stem

    # 1. Exact same filename.
    p = pred_dir / gt_path.name
    if p.exists():
        return p

    # 2. Same stem with common image suffixes.
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = pred_dir / f"{stem}{ext}"
        if p.exists():
            return p

    # 3. Recursive search with the same stem.
    candidates = [f for f in list_image_files(pred_dir) if f.stem == stem]

    # 4. Some prediction names may contain the original GT stem.
    if len(candidates) == 0:
        candidates = [f for f in list_image_files(pred_dir) if stem in f.stem]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No prediction found for GT: {gt_path.name} in {pred_dir}"
        )

    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def load_mask(mask_path, img_size, threshold):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    arr = np.array(mask)
    return (arr > threshold).astype(np.uint8)


def load_pred(pred_path, img_size, threshold):
    pred_img = Image.open(pred_path).convert("L")
    pred_img = pred_img.resize((img_size, img_size), Image.NEAREST)
    arr = np.array(pred_img).astype(np.float32)

    pred_binary = (arr > threshold).astype(np.uint8)

    # Used for AUC. If prediction is a binary mask, this is an approximate AUC
    # based on the binary result rather than a probability map.
    if arr.max() > arr.min():
        pred_score = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        pred_score = arr / 255.0

    return pred_binary, pred_score


def confusion(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp, tn, fp, fn


def dice_score(pred, gt, eps=1e-7):
    tp, _, fp, fn = confusion(pred, gt)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_score(pred, gt, eps=1e-7):
    tp, _, fp, fn = confusion(pred, gt)
    return (tp + eps) / (tp + fp + fn + eps)


def accuracy_score_binary(pred, gt, eps=1e-7):
    tp, tn, fp, fn = confusion(pred, gt)
    return (tp + tn + eps) / (tp + tn + fp + fn + eps)


def sensitivity_score(pred, gt, eps=1e-7):
    tp, _, _, fn = confusion(pred, gt)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(pred, gt, eps=1e-7):
    _, tn, fp, _ = confusion(pred, gt)
    return (tn + eps) / (tn + fp + eps)


def surface_distances(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None

    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b))

    dt_b = distance_transform_edt(~surface_b)
    dt_a = distance_transform_edt(~surface_a)

    dist_a_to_b = dt_b[surface_a]
    dist_b_to_a = dt_a[surface_b]

    return np.concatenate([dist_a_to_b, dist_b_to_a])


def hd95_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        h, w = gt.shape
        return float(np.sqrt(h * h + w * w))

    dists = surface_distances(pred, gt)
    if dists is None or len(dists) == 0:
        h, w = gt.shape
        return float(np.sqrt(h * h + w * w))

    return float(np.percentile(dists, 95))


def auc_score(pred_score, gt):
    gt_flat = gt.reshape(-1).astype(np.uint8)
    score_flat = pred_score.reshape(-1).astype(np.float32)

    if len(np.unique(gt_flat)) < 2:
        return np.nan

    return float(roc_auc_score(gt_flat, score_flat))


def evaluate_model(model_name, pred_dir, gt_files, img_size, threshold):
    rows = []
    all_scores = []
    all_gts = []

    for gt_path in tqdm(gt_files, desc=f"Evaluating {model_name}"):
        pred_path = find_prediction_file(pred_dir, gt_path)

        gt = load_mask(gt_path, img_size, threshold)
        pred, score = load_pred(pred_path, img_size, threshold)

        rows.append({
            "Model": model_name,
            "Image": gt_path.name,
            "Prediction": str(pred_path),
            "Dice": dice_score(pred, gt),
            "IoU": iou_score(pred, gt),
            "Accuracy": accuracy_score_binary(pred, gt),
            "Sensitivity": sensitivity_score(pred, gt),
            "Specificity": specificity_score(pred, gt),
            "AUC": auc_score(score, gt),
            "HD95": hd95_score(pred, gt),
        })

        all_scores.append(score.reshape(-1))
        all_gts.append(gt.reshape(-1))

    all_scores = np.concatenate(all_scores)
    all_gts = np.concatenate(all_gts).astype(np.uint8)

    if len(np.unique(all_gts)) >= 2:
        global_auc = float(roc_auc_score(all_gts, all_scores))
    else:
        global_auc = np.nan

    return rows, global_auc


def summarize(per_image_df, global_auc_dict):
    summary_rows = []

    for model_name, group in per_image_df.groupby("Model"):
        row = {"Model": model_name}

        for metric in ["Dice", "IoU", "Accuracy", "Sensitivity", "Specificity", "HD95"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        row["AUC_mean_per_image"] = group["AUC"].mean(skipna=True)
        row["AUC_std_per_image"] = group["AUC"].std(skipna=True)
        row["AUC_global"] = global_auc_dict.get(model_name, np.nan)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def parse_prediction_dirs(args):
    if args.pred_dirs:
        pred_dirs = {}
        for item in args.pred_dirs:
            if "=" not in item:
                raise ValueError(
                    f"Invalid --pred-dirs item: {item}. Use the format Name=Path."
                )
            name, path = item.split("=", 1)
            pred_dirs[name] = path
        return pred_dirs

    if args.pred_dir is None:
        raise ValueError("Either --pred-dir or --pred-dirs must be provided.")

    return {args.model_name: args.pred_dir}


def main():
    args = parse_args()

    gt_dir = Path(args.gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    gt_files = list_image_files(gt_dir)
    if len(gt_files) == 0:
        raise RuntimeError(f"No GT mask files found in: {gt_dir}")

    pred_dirs = parse_prediction_dirs(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Evaluate binary segmentation results from prediction masks")
    print(f"GT mask dir: {gt_dir}")
    print(f"Number of GT masks: {len(gt_files)}")
    print("=" * 100)

    all_rows = []
    global_auc_dict = {}

    for model_name, pred_dir in pred_dirs.items():
        pred_dir = Path(pred_dir)
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

        print("\n" + "=" * 100)
        print(f"Model: {model_name}")
        print(f"Prediction dir: {pred_dir}")

        rows, global_auc = evaluate_model(
            model_name=model_name,
            pred_dir=pred_dir,
            gt_files=gt_files,
            img_size=args.img_size,
            threshold=args.threshold,
        )

        all_rows.extend(rows)
        global_auc_dict[model_name] = global_auc

    per_image_df = pd.DataFrame(all_rows)
    summary_df = summarize(per_image_df, global_auc_dict)

    per_image_path = out_dir / f"{args.out_prefix}_per_image_metrics.csv"
    summary_path = out_dir / f"{args.out_prefix}_summary_metrics.csv"

    per_image_df.to_csv(per_image_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Finished.")
    print(f"Saved per-image metrics: {per_image_path}")
    print(f"Saved summary metrics: {summary_path}")

    show_cols = [
        "Model",
        "Dice_mean",
        "IoU_mean",
        "Accuracy_mean",
        "Sensitivity_mean",
        "Specificity_mean",
        "AUC_global",
        "HD95_mean",
    ]

    print("\nSummary:")
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
