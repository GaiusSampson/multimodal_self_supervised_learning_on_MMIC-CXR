import os
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.transforms import Compose, Resize, InterpolationMode
from torchvision.transforms.functional import to_pil_image

# Project helpers
from train import load_clip
from zero_shot_flava import (
    run_softmax_eval_flava,
    make_true_labels,
    CXRTestDatasetFLAVA,
    load_test_data_flava
)
from eval import evaluate, bootstrap
from metrics import compute_f1, compute_mcc, get_best_p_vals

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_biobert', action='store_true',
                            help='Use BioClinicalBERT for text encoder')
    args = parser.parse_args()
    return args

args = parse_args()
# ===================== CONFIG =====================
# Test set
cxr_filepath: str = './test_data/chexpert_test.h5'
cxr_true_labels_path: Optional[str] = './test_data/groundtruth.csv'

# Validation set (for threshold tuning)
val_img_path: str = "./test_data/chexpert_val.h5"
val_label_path: str = "./test_data/val.csv"

# Checkpoints root
model_dir: str = './checkpoints/pt-imp'

# Output dirs
predictions_dir: Path = Path('./predictions')
cache_dir: Path = predictions_dir / "cached"
predictions_dir.mkdir(exist_ok=True, parents=True)
cache_dir.mkdir(exist_ok=True, parents=True)

# Model configuration
use_biobert: bool = args.use_biobert
biobert_model: str = "emilyalsentzer/Bio_ClinicalBERT"
context_length: int = 128

batch_size: int = 1
num_workers: int = 0

# GPU toggle
USE_GPU = True and torch.cuda.is_available()
if USE_GPU:
    torch.backends.cudnn.benchmark = True

# Labels & prompts
cxr_labels: List[str] = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]
cxr_pair_template: Tuple[str, str] = ("{}", "no {}")

# ==================================================


# ============== Simple checkpoint collection ==============
def collect_model_paths(root: str) -> List[str]:
    """Recursively collect all .pt checkpoint files"""
    paths = []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.endswith('.pt'):
                paths.append(os.path.join(subdir, f))
    return sorted(paths)


# ================= GPU Adapter =================
class GPUAdapter(nn.Module):
    """
    Wraps FLAVA model so encode_image/encode_text run on GPU
    but return CPU tensors for downstream numpy code.
    Mirrors the GPUAdapter in the Swin test script.
    """
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.base = base_model.to(device).eval()
        self.device = device

        # Copy FLAVA-specific attributes
        for attr in ('use_biobert', 'use_flava', 'tokenizer',
                     'context_length', 'processor'):
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.encode_image(x.to(self.device, non_blocking=True))
        return out.detach().cpu()

    @torch.no_grad()
    def encode_text(self, t) -> torch.Tensor:
        if isinstance(t, dict):
            t = {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
        else:
            t = t.to(self.device, non_blocking=True)
        out = self.base.encode_text(t)
        return out.detach().cpu()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


# ================= FLAVA model loader =================
def make_flava(
    model_path: str,
    cxr_h5: str,
    use_biobert: bool = True,
    biobert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    context_length: int = 128,
) -> Tuple[nn.Module, DataLoader]:
    """
    Build FLAVA model, load fine-tuned weights, and create data loader.

    Args:
        model_path: Path to .pt checkpoint file
        cxr_h5: Path to HDF5 image file
        use_biobert: Whether to use BioClinicalBERT text encoder
        biobert_model: Which BioBERT variant to use
        context_length: Max token sequence length

    Returns:
        model, loader tuple
    """
    # Build FLAVA model architecture
    base = load_clip(
        model_path=None,
        pretrained=False,
        context_length=context_length,
        flava=True,
        use_biobert=use_biobert,
        biobert_model=biobert_model
    )

    def load_any_state_dict(path, device='cpu'):
        """Load checkpoint handling both raw state_dict and wrapped checkpoints"""
        # Add numpy scalar to safe globals for PyTorch 2.6+
        try:
            import numpy._core.multiarray
            torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        except (AttributeError, ImportError):
            pass  # Older PyTorch or numpy versions

        try:
            obj = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            # Fall back to weights_only=False for legacy checkpoints
            obj = torch.load(path, map_location=device, weights_only=False)

        if isinstance(obj, dict) and "model_state_dict" in obj:
            return obj["model_state_dict"]
        return obj

    # Load weights
    state = load_any_state_dict(model_path, device='cpu')
    missing_unexpected = base.load_state_dict(state, strict=False)

    print(f"Loading checkpoint: {model_path}")
    if missing_unexpected.missing_keys:
        print(f"  Missing keys:    {missing_unexpected.missing_keys}")
    if missing_unexpected.unexpected_keys:
        print(f"  Unexpected keys: {missing_unexpected.unexpected_keys}")

    base.eval()

    # Wrap with GPU adapter if available
    if USE_GPU:
        device = torch.device("cuda")
        model = GPUAdapter(base, device)
        pin = True
        print("Using GPU acceleration")
    else:
        model = base
        pin = False
        print("Using CPU")

    # Create FLAVA-appropriate dataset and loader
    dset = CXRTestDatasetFLAVA(img_path=cxr_h5)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    return model, loader


# ================= Ensemble =================
def ensemble_models(
    model_paths: List[str],
    cxr_h5: str,
    cxr_labels: List[str],
    cxr_pair_template: Tuple[str, str],
    use_biobert: bool,
    biobert_model: str,
    context_length: int,
    cache_dir: Optional[Path] = None,
    save_name: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Evaluate multiple checkpoints and ensemble their predictions.

    Args:
        model_paths: List of checkpoint paths
        cxr_h5: Path to HDF5 image file
        cxr_labels: List of pathology labels
        cxr_pair_template: Positive/negative template pair
        use_biobert: Whether to use BioClinicalBERT
        biobert_model: Which BioBERT variant to use
        context_length: Max token sequence length
        cache_dir: Directory to cache predictions
        save_name: Optional prefix for cache files

    Returns:
        List of individual predictions and averaged ensemble prediction
    """
    predictions = []

    for path in model_paths:
        model_name = Path(path).stem

        # Check for cached prediction
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_file = (
                f"{save_name}_{model_name}.npy"
                if save_name else f"{model_name}.npy"
            )
            cache_path = cache_dir / cache_file

        if cache_path is not None and cache_path.exists():
            print(f"Loading cached prediction for {model_name}")
            y_pred = np.load(cache_path)
        else:
            print(f"\nInferring model: {path}")

            model, loader = make_flava(
                model_path=path,
                cxr_h5=cxr_h5,
                use_biobert=use_biobert,
                biobert_model=biobert_model,
                context_length=context_length
            )

            with torch.no_grad():
                y_pred = run_softmax_eval_flava(
                    model, loader, cxr_labels, cxr_pair_template
                )

            # Cache prediction
            if cache_path is not None:
                np.save(cache_path, y_pred)
                print(f"Cached prediction to {cache_path}")

        predictions.append(y_pred)

    y_pred_avg = np.mean(predictions, axis=0)
    return predictions, y_pred_avg


# ================= Main =================
if __name__ == "__main__":
    print("=" * 60)
    print("Zero-Shot Evaluation: FLAVA")
    print("=" * 60)
    print(f"Text Encoder:   {'BioClinicalBERT (' + biobert_model + ')' if use_biobert else 'Standard BERT'}")
    print(f"Context Length: {context_length}")
    print(f"GPU Enabled:    {USE_GPU}")
    print("=" * 60)

    # Collect checkpoints
    model_paths: List[str] = collect_model_paths(model_dir)
    print(f"\nFound {len(model_paths)} checkpoint(s):")
    for p in model_paths:
        print(f"  - {p}")

    if len(model_paths) == 0:
        print(f"\nERROR: No checkpoints found in {model_dir}")
        print("Please check model_dir and ensure .pt files exist.")
        sys.exit(1)


    print("\n" + "=" * 60)
    print("Running Test Set Evaluation")
    print("=" * 60)

    preds_list, y_pred_avg = ensemble_models(
        model_paths=model_paths,
        cxr_h5=cxr_filepath,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        use_biobert=use_biobert,
        biobert_model=biobert_model,
        context_length=context_length,
        cache_dir=cache_dir,
    )

    pred_path = predictions_dir / "chexpert_preds_flava_ensemble.npy"
    np.save(pred_path, y_pred_avg)
    print(f"\n Saved ensemble predictions to {pred_path}")

    # AUC on test set
    print("\n" + "=" * 60)
    print("Computing AUC Metrics")
    print("=" * 60)

    test_true = make_true_labels(
        cxr_true_labels_path=cxr_true_labels_path,
        cxr_labels=cxr_labels
    )
    auc_df = evaluate(y_pred_avg, test_true, cxr_labels)
    print("\nPer-label AUCs:")
    print(auc_df)

    mean_auc = auc_df.mean(axis=1)[0]
    print(f"\n Mean AUC: {mean_auc:.4f}")

    # Bootstrap 95% CI on test set
    print("\n" + "=" * 60)
    print("Computing Bootstrap Confidence Intervals")
    print("=" * 60)

    auc_ci_df = bootstrap(y_pred_avg, test_true, cxr_labels)[1]
    print("\nAUC with 95% CI:")
    print(auc_ci_df)

    auc_ci_csv = predictions_dir / "chexpert_auc_flava.csv"
    auc_ci_df.to_csv(auc_ci_csv, index=True)
    print(f"\n Saved AUC with CI to: {auc_ci_csv}")

    # F1 & MCC using val thresholds 
    if os.path.exists(val_img_path) and os.path.exists(val_label_path):
        print("\n" + "=" * 60)
        print("Computing F1 and MCC with Validation Thresholds")
        print("=" * 60)

        try:
            val_true = make_true_labels(val_label_path, cxr_labels)

            _, val_pred_avg = ensemble_models(
                model_paths=model_paths,
                cxr_h5=val_img_path,
                cxr_labels=cxr_labels,
                cxr_pair_template=cxr_pair_template,
                use_biobert=use_biobert,
                biobert_model=biobert_model,
                context_length=context_length,
                cache_dir=cache_dir,
                save_name="val"
            )

            # Optimal thresholds from validation set
            best_p_vals = get_best_p_vals(val_pred_avg, val_true, cxr_labels)

            # F1 and MCC on test set
            f1_ci  = compute_f1(y_pred_avg, test_true, cxr_labels, best_p_vals)
            mcc_ci = compute_mcc(y_pred_avg, test_true, cxr_labels, best_p_vals)

            f1_csv  = predictions_dir / "chexpert_f1_flava.csv"
            mcc_csv = predictions_dir / "chexpert_mcc_flava.csv"
            f1_ci.to_csv(f1_csv, index=True)
            mcc_ci.to_csv(mcc_csv, index=True)

            print("\nF1 Scores with 95% CI:")
            print(f1_ci)
            print("\nMCC Scores with 95% CI:")
            print(mcc_ci)
            print(f"\n Saved F1  with CI to: {f1_csv}")
            print(f" Saved MCC with CI to: {mcc_csv}")

        except Exception as e:
            print(f"\nWarning: Could not compute F1/MCC metrics: {e}")
            import traceback
            traceback.print_exc()


    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {predictions_dir}")
    print(f"  - Predictions:  {pred_path.name}")
    print(f"  - AUC with CI:  {auc_ci_csv.name}")
    if os.path.exists(val_img_path):
        print(f"  - F1 scores:    chexpert_f1_flava.csv")
        print(f"  - MCC scores:   chexpert_mcc_flava.csv")