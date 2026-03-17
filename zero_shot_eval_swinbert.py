import os
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.append('../') 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, Lambda

# Your project helpers
from train import load_clip as load_swin_biobert_clip
from zero_shot_biobert import run_softmax_eval, make_true_labels
from eval import evaluate, bootstrap

from metrics import compute_f1, compute_mcc, get_best_p_vals

# timm for eval transform
from timm.data import resolve_data_config

# ===================== CONFIG =====================
# Test set
cxr_filepath: str = './test_data/chexpert_test.h5'
cxr_true_labels_path: Optional[str] = './test_data/groundtruth.csv'

# Validation set (for threshold tuning)
val_img_path = "./test_data/chexpert_val.h5"
val_label_path = "./test_data/val.csv"

# Checkpoints root (all .pt under here will be evaluated/ensembled)
model_dir: str = './checkpoints/pt-imp'

# Output dirs
predictions_dir: Path = Path('./predictions')
cache_dir: Path = predictions_dir / "cached"
predictions_dir.mkdir(exist_ok=True, parents=True)
cache_dir.mkdir(exist_ok=True, parents=True)

# Model configuration
use_swin_encoder: bool = True
use_biobert: bool = True
biobert_model: str = "emilyalsentzer/Bio_ClinicalBERT"
context_length: int = 77

batch_size: int = 1
num_workers: int = 0 

# GPU toggle
USE_GPU = True and torch.cuda.is_available()
if USE_GPU:
    torch.backends.cudnn.benchmark = True

# Labels & prompts
cxr_labels: List[str] = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema',
    'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
    'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
    'Pneumothorax','Support Devices'
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


# ================= Dataset (HDF5 -> tensor) =================
class CXRTestDataset(Dataset):
    """Dataset for loading CXR images from HDF5 file"""
    def __init__(self, img_path: str, transform=None):
        self.f = h5py.File(img_path, 'r')
        self.imgs = self.f['cxr']
        self.transform = transform
        
    def __len__(self): 
        return len(self.imgs)
    
    def __getitem__(self, idx):
        x = self.imgs[idx]                 # (H, W) grayscale
        x = np.repeat(x[None, ...], 3, 0)  # (3, H, W)
        x = torch.from_numpy(x).float()
        return {'img': self.transform(x) if self.transform else x}
    
    def __del__(self):
        try: 
            self.f.close()
        except: 
            pass


# ============== timm-driven eval transform ==============
def _get_backbone_for_timm(model):
    """Extract the vision backbone from the model"""
    # Try common attribute names used in CLIP wrappers
    for attr in ("image_encoder", "visual", "vision_model", "image_model", "backbone"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def get_timm_eval_transform_from_model(model):
    
    backbone = _get_backbone_for_timm(model)
    cfg = resolve_data_config({}, model=backbone)  # input_size, mean, std, interpolation, crop_pct

    H, W = cfg["input_size"][1], cfg["input_size"][2]
    crop_pct = cfg.get("crop_pct", 1.0)
    interp_str = (cfg.get("interpolation", "bicubic") or "bicubic").lower()
    interp_map = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    interp = interp_map.get(interp_str, InterpolationMode.BICUBIC)

    # Pre-crop resize size as timm does
    if crop_pct and crop_pct < 1.0:
        resize_h = int(math.floor(H / crop_pct))
        resize_w = int(math.floor(W / crop_pct))
    else:
        resize_h, resize_w = H, W

    return Compose([
        Lambda(lambda t: t / 255.0),                    # to [0,1]
        Resize((resize_h, resize_w), interpolation=interp),
        CenterCrop((H, W)),
        Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])



class GPUAdapter(nn.Module):
    """
    Wraps your CLIP model so encode_image/encode_text run on GPU,
    but return CPU tensors for downstream numpy code.
    Also forwards attribute access to handle BioBERT-specific attributes.
    """
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.base = base_model.to(device).eval()
        self.device = device
        
        # Copy important attributes from base model
        if hasattr(base_model, 'use_biobert'):
            self.use_biobert = base_model.use_biobert
        if hasattr(base_model, 'tokenizer'):
            self.tokenizer = base_model.tokenizer
        if hasattr(base_model, 'context_length'):
            self.context_length = base_model.context_length

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.encode_image(x.to(self.device, non_blocking=True))
        return out.detach().cpu()

    @torch.no_grad()
    def encode_text(self, t):
        """
        Handle both standard CLIP tokens (tensor) and BioBERT tokens (dict)
        """
        if isinstance(t, dict):
            # BioBERT: move all dict values to device
            t = {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
        else:
            # Standard CLIP: move tensor to device
            t = t.to(self.device, non_blocking=True)
        
        out = self.base.encode_text(t)
        return out.detach().cpu()

    def __getattr__(self, name):
        if name in {"base", "device", "encode_image", "encode_text", "use_biobert", "tokenizer", "context_length"}:
            return super().__getattr__(name)
        return getattr(self.base, name)


# ================= Swin+BioBERT CLIP loader =================
def make_swin_biobert(
    model_path: str, 
    cxr_h5: str, 
    context_length: int,
    use_swin: bool = True,
    use_biobert: bool = True,
    biobert_model: str = "emilyalsentzer/Bio_ClinicalBERT"
):
    """
    Build Swin+BioBERT CLIP model and load fine-tuned weights.
    Keeps DataLoader batch_size=1 so original zero_shot.predict() works.
    
    Args:
        model_path: Path to checkpoint file
        cxr_h5: Path to HDF5 image file
        context_length: Max sequence length
        use_swin: Use Swin Transformer for vision
        use_biobert: Use BioClinicalBERT for text
        biobert_model: Which BioBERT variant to use
    """
    # Build model architecture (no weights yet)
    base = load_swin_biobert_clip(
        model_path=None,
        context_length=context_length,
        pretrained=False,
        swin_encoder=use_swin,
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
        print("Missing keys:", missing_unexpected.missing_keys)
    if missing_unexpected.unexpected_keys:
        print("Unexpected keys:", missing_unexpected.unexpected_keys)

    base.eval()

    # Get appropriate transform
    transform = get_timm_eval_transform_from_model(base)

    # Optional GPU wrapping (compute on GPU, return CPU tensors)
    if USE_GPU:
        device = torch.device("cuda")
        model = GPUAdapter(base, device)
        pin = True
        print("Using GPU acceleration")
    else:
        model = base
        pin = False
        print("Using CPU")

    # Create dataset and loader
    dset = CXRTestDataset(cxr_h5, transform=transform)
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
    context_length: int,
    use_swin: bool,
    use_biobert: bool,
    biobert_model: str,
    cache_dir: Optional[Path] = None,
    save_name: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Evaluate multiple checkpoints and ensemble their predictions.
    
    Returns:
        List of individual predictions and averaged ensemble prediction
    """
    predictions = []
    
    for path in model_paths:
        model_name = Path(path).stem

        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_file = f"{save_name}_{model_name}.npy" if save_name else f"{model_name}.npy"
            cache_path = cache_dir / cache_file

        # Load cached prediction if available
        if cache_path is not None and cache_path.exists():
            print(f"Loading cached prediction for {model_name}")
            y_pred = np.load(cache_path)
        else:
            print(f"\nInferring model: {path}")
            
            # Load model and data
            model, loader = make_swin_biobert(
                path, cxr_h5, context_length,
                use_swin=use_swin,
                use_biobert=use_biobert,
                biobert_model=biobert_model
            )
            
            # Run evaluation
            with torch.no_grad():
                y_pred = run_softmax_eval(
                    model, loader, cxr_labels, cxr_pair_template,
                    context_length=context_length
                )
            
            # Cache prediction
            if cache_path is not None:
                np.save(cache_path, y_pred)
                print(f"Cached prediction to {cache_path}")

        predictions.append(y_pred)

    # Ensemble by averaging
    y_pred_avg = np.mean(predictions, axis=0)
    return predictions, y_pred_avg



if __name__ == "__main__":
    print("="*60)
    print("Zero-Shot Evaluation: Swin Transformer + BioClinicalBERT")
    print("="*60)
    print(f"Vision Encoder: {'Swin Transformer V2' if use_swin_encoder else 'Standard ViT'}")
    print(f"Text Encoder: {biobert_model if use_biobert else 'Standard CLIP'}")
    print(f"Context Length: {context_length}")
    print(f"GPU Enabled: {USE_GPU}")
    print("="*60)
    
    # Collect checkpoints
    model_paths: List[str] = collect_model_paths(model_dir)
    print(f"\nFound {len(model_paths)} checkpoint(s):")
    for p in model_paths:
        print(f"  - {p}")
    
    if len(model_paths) == 0:
        print(f"\nERROR: No checkpoints found in {model_dir}")
        print("Please check the model_dir path and ensure .pt files exist.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Running Test Set Evaluation")
    print("="*60)
    
    # 1) Test-set predictions (ensemble)
    preds_list, y_pred_avg = ensemble_models(
        model_paths=model_paths,
        cxr_h5=cxr_filepath,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        context_length=context_length,
        use_swin=use_swin_encoder,
        use_biobert=use_biobert,
        biobert_model=biobert_model,
        cache_dir=cache_dir,
    )

    # 2) Save averaged predictions
    pred_path = predictions_dir / "chexpert_preds_swin_biobert_ensemble.npy"
    np.save(pred_path, y_pred_avg)
    print(f"\n✓ Saved ensemble predictions to {pred_path}")

    # 3) AUC (per-label) on TEST
    print("\n" + "="*60)
    print("Computing AUC Metrics")
    print("="*60)
    
    test_true = make_true_labels(
        cxr_true_labels_path=cxr_true_labels_path, 
        cxr_labels=cxr_labels
    )
    auc_df = evaluate(y_pred_avg, test_true, cxr_labels)
    print("\nPer-label AUCs:")
    print(auc_df)
    
    mean_auc = auc_df.mean(axis=1)[0]
    print(f"\n✓ Mean AUC: {mean_auc:.4f}")

    # 4) AUC 95% CI (bootstrap) on TEST
    print("\n" + "="*60)
    print("Computing Bootstrap Confidence Intervals")
    print("="*60)
    
    auc_ci_df = bootstrap(y_pred_avg, test_true, cxr_labels)[1]
    print("\nAUC with 95% CI:")
    print(auc_ci_df)
    
    auc_ci_csv = predictions_dir / "chexpert_auc_swin_biobert.csv"
    auc_ci_df.to_csv(auc_ci_csv, index=True)
    print(f"\n✓ Saved AUC with CI to: {auc_ci_csv}")

    # 5) Thresholds from VAL → F1 & MCC on TEST
    if os.path.exists(val_img_path) and os.path.exists(val_label_path):
        print("\n" + "="*60)
        print("Computing F1 and MCC with Validation Thresholds")
        print("="*60)
        
        try:
            # Get validation predictions
            val_true = make_true_labels(val_label_path, cxr_labels)
            _, val_pred_avg = ensemble_models(
                model_paths=model_paths,
                cxr_h5=val_img_path,
                cxr_labels=cxr_labels,
                cxr_pair_template=cxr_pair_template,
                context_length=context_length,
                use_swin=use_swin_encoder,
                use_biobert=use_biobert,
                biobert_model=biobert_model,
                cache_dir=cache_dir,
                save_name="val"
            )
            
            # Get optimal thresholds from validation set
            best_p_vals = get_best_p_vals(val_pred_avg, val_true, cxr_labels)
            
            # Compute F1 and MCC on test set with those thresholds
            f1_ci = compute_f1(y_pred_avg, test_true, cxr_labels, best_p_vals)
            mcc_ci = compute_mcc(y_pred_avg, test_true, cxr_labels, best_p_vals)
            
            # Save results
            f1_csv = predictions_dir / "chexpert_f1_swin_biobert.csv"
            mcc_csv = predictions_dir / "chexpert_mcc_swin_biobert.csv"
            f1_ci.to_csv(f1_csv, index=True)
            mcc_ci.to_csv(mcc_csv, index=True)
            
            print("\nF1 Scores with 95% CI:")
            print(f1_ci)
            print("\nMCC Scores with 95% CI:")
            print(mcc_ci)
            
            print(f"\nSaved F1 with CI to:  {f1_csv}")
            print(f"\nSaved MCC with CI to: {mcc_csv}")
            
        except Exception as e:
            print(f"\nWarning: Could not compute F1/MCC metrics: {e}")

    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults saved to: {predictions_dir}")
    print(f"  - Predictions: {pred_path.name}")
    print(f"  - AUC with CI: {auc_ci_csv.name}")
    if os.path.exists(val_img_path):
        print(f"  - F1 scores: chexpert_f1_swin_biobert.csv")
        print(f"  - MCC scores: chexpert_mcc_swin_biobert.csv")