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


from train import load_clip as load_swin_clip
from zero_shot import run_softmax_eval, make_true_labels
from eval import evaluate, bootstrap
from metrics import compute_f1, compute_mcc, get_best_p_vals


from timm.data import resolve_data_config

# ===================== CONFIG =====================
# Test set
cxr_filepath: str = './test_data/chexpert_test.h5'
cxr_true_labels_path: Optional[str] = './test_data/groundtruth.csv'

# Validation set (for threshold tuning)
val_img_path = "./test_data/chexpert_val.h5"
val_label_path = "./test_data/val.csv"

# Checkpoints root 
model_dir: str = './checkpoints'

# Output dirs
predictions_dir: Path = Path('./predictions')
cache_dir: Path = predictions_dir / "cached"
predictions_dir.mkdir(exist_ok=True, parents=True)
cache_dir.mkdir(exist_ok=True, parents=True)

# Eval hyperparams
context_length: int = 77

# Important: keep batch_size=1 so original zero_shot.predict() (which squeezes dim-0) works.
batch_size: int = 1
num_workers: int = 0  # safer for h5py

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

# cxr_pair_template: Tuple[str] = (
#     "There is evidence of {}.",
#     "No signs of {} are seen.",
#     "A chest X-ray showing {}.",
#     "Findings suggestive of {}.",
#     "There is {}.",
#     "Absence of {}.",
#     "Findings consistent with {}.",
#     "The chest X-ray shows {}.",
#     "Presence of {}.",
#     "No radiographic evidence of {}.",
#     "There is no {} present."
# )

# ==================================================

# ============== Simple checkpoint collection ==============
def collect_model_paths(root: str) -> List[str]:
    paths = []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.endswith('.pt'):
                paths.append(os.path.join(subdir, f))
    return sorted(paths)

model_paths: List[str] = collect_model_paths(model_dir)
print(f"Found {len(model_paths)} checkpoints")
for p in model_paths:
    print(" -", p)


class CXRTestDataset(Dataset):
    def __init__(self, img_path: str, transform=None):
        self.f = h5py.File(img_path, 'r')
        self.imgs = self.f['cxr']
        self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        x = self.imgs[idx]                
        x = np.repeat(x[None, ...], 3, 0)  
        x = torch.from_numpy(x).float()
        return {'img': self.transform(x) if self.transform else x}
    def __del__(self):
        try: self.f.close()
        except: pass

# ============== timm-driven eval transform ==============
def _get_backbone_for_timm(model):
    for attr in ("image_encoder", "visual", "vision_model", "image_model", "backbone"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model

def get_timm_eval_transform_from_model(model):
    """
    Build a transform matching timm's eval pipeline using the model's config.
    Works on tensor inputs (CxHxW). Assumes tensor is in [0,255].
    """
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

# ============== GPU adapter  ==============
class GPUAdapter(nn.Module):
    """
    Wraps the CLIP model so encode_image/encode_text run on GPU,
    but return CPU tensors for downstream numpy code.
    """
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.base = base_model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.encode_image(x.to(self.device, non_blocking=True))
        return out.detach().cpu()

    @torch.no_grad()
    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        out = self.base.encode_text(t.to(self.device, non_blocking=True))
        return out.detach().cpu()

    def __getattr__(self, name):
        if name in {"base", "device", "encode_image", "encode_text"}:
            return super().__getattr__(name)
        return getattr(self.base, name)

# ================= Swin-CLIP loader =================
def make_swin(model_path: str, cxr_h5: str, context_length: int):
    """
    Rebuild your Swin-CLIP and load fine-tuned weights.
    Keeps DataLoader batch_size=1 so original zero_shot.predict() works.
    """
    # Build + load weights on CPU
    base = load_swin_clip(
        model_path=None,
        context_length=context_length,
        pretrained=False,      # load fine-tuned weights next
        swin_encoder=True
    )
    
    def load_any_state_dict(path, device='cpu'):
        try:
            obj = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location=device)
        # unwrap full checkpoint if needed
        if isinstance(obj, dict) and "model_state_dict" in obj:
            return obj["model_state_dict"]
        return obj

    state = load_any_state_dict(model_path, device='cpu')


    missing_unexpected = base.load_state_dict(state, strict=False)
    print("Missing:", missing_unexpected.missing_keys)
    print("Unexpected:", missing_unexpected.unexpected_keys)

    base.eval()


    transform = get_timm_eval_transform_from_model(base)

    # Optional GPU wrapping (compute on GPU, return CPU tensors)
    if USE_GPU:
        device = torch.device("cuda")
        model = GPUAdapter(base, device)
        pin = True
    else:
        model = base
        pin = False

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
    cache_dir: Optional[Path] = None,
    save_name: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:

    predictions = []
    for path in model_paths:
        model_name = Path(path).stem

        # Cache (optional)
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_file = f"{save_name}_{model_name}.npy" if save_name else f"{model_name}.npy"
            cache_path = cache_dir / cache_file

        if cache_path is not None and cache_path.exists():
            print(f"Loading cached prediction for {model_name}")
            y_pred = np.load(cache_path)
        else:
            print(f"Inferring model {path}")
            model, loader = make_swin(path, cxr_h5, context_length)
            with torch.no_grad():
                y_pred = run_softmax_eval(
                    model, loader, cxr_labels, cxr_pair_template,
                    context_length=context_length
                )
            if cache_path is not None:
                np.save(cache_path, y_pred)

        predictions.append(y_pred)

    y_pred_avg = np.mean(predictions, axis=0)
    return predictions, y_pred_avg

# ================= RUN =================
if __name__ == "__main__":
    # 1) Test-set predictions (ensemble)
    preds_list, y_pred_avg = ensemble_models(
        model_paths=model_paths,
        cxr_h5=cxr_filepath,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        cache_dir=cache_dir,
    )

    # 2) Save averaged preds
    pred_path = predictions_dir / "chexpert_preds_swin_ensemble_all_best_ckpts_re.npy"
    """np.save(pred_path, y_pred_avg)"""
    print(f"Saved ensemble preds to {pred_path}")

    # 3) AUC (per-label) on TEST
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
    auc_df = evaluate(y_pred_avg, test_true, cxr_labels)
    print("Per-label AUCs:\n", auc_df)

    # 4) AUC 95% CI (bootstrap) on TEST
    auc_ci_df = bootstrap(y_pred_avg, test_true, cxr_labels)[1]
    print("AUC with 95% CI:\n", auc_ci_df)
    auc_ci_csv = predictions_dir / "chexpert_auc_swin_best_ckpts.csv"
    auc_ci_df.to_csv(auc_ci_csv, index=True)
    print(f"Saved AUC CI to: {auc_ci_csv}")

    # 5) Thresholds from VAL → F1 & MCC on TEST
    val_true = make_true_labels(val_label_path, cxr_labels)
    _, val_pred_avg = ensemble_models(
        model_paths=model_paths,
        cxr_h5=val_img_path,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        cache_dir=cache_dir,
        save_name="val"
    )
    best_p_vals = get_best_p_vals(val_pred_avg, val_true, cxr_labels)

    f1_ci  = compute_f1(y_pred_avg, test_true, cxr_labels, best_p_vals)
    mcc_ci = compute_mcc(y_pred_avg, test_true, cxr_labels, best_p_vals)

    f1_csv  = predictions_dir / "chexpert_f1_swin_best_ckpts.csv"
    mcc_csv = predictions_dir / "chexpert_mcc_swin_best_ckpts.csv"
    f1_ci.to_csv(f1_csv, index=True)
    mcc_ci.to_csv(mcc_csv, index=True)
    print(f"Saved F1 CI to:  {f1_csv}")
    print(f"Saved MCC CI to: {mcc_csv}")
