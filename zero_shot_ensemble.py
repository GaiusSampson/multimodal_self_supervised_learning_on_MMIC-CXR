import os
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict#
import argparse

sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, Lambda

# Your project helpers
from zero_shot import make_true_labels
from eval import evaluate, bootstrap
from metrics import compute_f1, compute_mcc, get_best_p_vals

# timm for Swin transforms
try:
    from timm.data import resolve_data_config
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not found. Swin models may not work optimally.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ensemble_strategy',
        type=str,
        default='mean',
        choices=['mean', 'uncertainty', 'rank', 'agreement', 'paper'],
        help='Ensemble strategy: mean (simple average), uncertainty (confidence-weighted), median (robust)'
    )

    return parser.parse_args()
 
args = parse_args()

# ===================== CONFIG =====================
# Test set
cxr_filepath: str = './test_data/chexpert_test.h5'
cxr_true_labels_path: Optional[str] = './test_data/groundtruth.csv'

# Validation set (for threshold tuning)
val_img_path = "./test_data/chexpert_val.h5"
val_label_path = "./test_data/val.csv"

# The directory for the checkpoints
checkpoint_dir = "best_models"

# The subdirectories for each model (can be empty if not using that architecture)
checkpoint_dirs: Dict[str, str] = {
    'vit': f'./{checkpoint_dir}/vit',              # ViT-B/32 models
    'resnet': f'./{checkpoint_dir}/resnet',        # ResNet-50 models
    'swin': f'./{checkpoint_dir}/swin',            # Swin Transformer V2 models
    'swin_biobert': f'./{checkpoint_dir}/swin_biobert',  # Swin + BioBERT models
    'flava': f'./{checkpoint_dir}/flava',          # FLAVA models (standard BERT)
    'flava_biobert': f'./{checkpoint_dir}/flava_biobert'  # FLAVA + BioBERT models
}

# Output dirs
predictions_dir: Path = Path('./predictions_universal_ensemble')
cache_dir: Path = predictions_dir / "cached"
predictions_dir.mkdir(exist_ok=True, parents=True)
cache_dir.mkdir(exist_ok=True, parents=True)

# Eval hyperparams
batch_size: int = 1  # Keep at 1 for compatibility
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

def collect_model_paths_by_architecture(checkpoint_dirs: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Collect checkpoint paths from architecture-specific subdirectories.
    """
    architecture_checkpoints = {}

    for arch_name, arch_dir in checkpoint_dirs.items():
        paths = []

        if not os.path.exists(arch_dir):
            print(f"Directory not found: {arch_dir} (skipping {arch_name})")
            architecture_checkpoints[arch_name] = []
            continue

        for subdir, _, files in os.walk(arch_dir):
            for f in files:
                if f.endswith('.pt'):
                    paths.append(os.path.join(subdir, f))

        architecture_checkpoints[arch_name] = sorted(paths)

        if len(paths) > 0:
            print(f"Found {len(paths)} {arch_name} checkpoint(s)")
        else:
            print(f"  No checkpoints in {arch_dir}")

    return architecture_checkpoints


class CXRTestDataset(Dataset):
    """Dataset for loading CXR images from HDF5"""
    def __init__(self, img_path: str, transform=None):
        self.f = h5py.File(img_path, 'r')
        self.imgs = self.f['cxr']
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = self.imgs[idx]                  # (H, W) grayscale
        x = np.repeat(x[None, ...], 3, 0)  # (3, H, W)
        x = torch.from_numpy(x).float()
        return {'img': self.transform(x) if self.transform else x}

    def __del__(self):
        try:
            self.f.close()
        except:
            pass


class CXRTestDatasetFLAVA(Dataset):
    """
    CXR test dataset for FLAVA models.
    Returns images scaled to [0, 1] without normalization
    since FLAVA's processor handles its own normalization.
    """
    def __init__(self, img_path: str):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']

    def __len__(self):
        return len(self.img_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx]  # np array (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float()

        # Scale to [0, 1] — FLAVA processor handles normalization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Resize to 224 for FLAVA's ViT
        resize = Resize(224, interpolation=InterpolationMode.BICUBIC)
        img = resize(img)

        return {'img': img}

def uncertainty_weighted_ensemble(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Weight models by their confidence/certainty.
    
    Args:
        predictions_list: List of prediction arrays [num_models, num_samples, num_classes]
    
    Returns:
        Ensemble predictions [num_samples, num_classes]
    """
    predictions_array = np.array(predictions_list)  # [num_models, num_samples, num_classes]
    num_models, num_samples, num_classes = predictions_array.shape
    
    print(f"\nApplying uncertainty weighting to {num_models} models...")
    print(f"Computing confidence-based weights for {num_samples} samples x {num_classes} classes")
    
    ensemble = np.zeros((num_samples, num_classes))
    
    # Track average weights per model for reporting
    model_weight_sums = np.zeros(num_models)
    total_decisions = num_samples * num_classes
    
    for sample_idx in range(num_samples):
        for class_idx in range(num_classes):
            # Get predictions from all models for this sample+class
            preds = predictions_array[:, sample_idx, class_idx]
            
            # Compute confidence: distance from 0.5 (uncertain)
            # Prediction of 0.9 or 0.1 is confident (distance = 0.4)
            # Prediction of 0.5 is uncertain (distance = 0.0)
            confidences = np.abs(preds - 0.5)
            
            # Convert to weights (more confident = higher weight)
            # Add small epsilon to avoid division by zero
            weights = confidences + 1e-6
            weights = weights / weights.sum()
            
            # Track weights for reporting
            model_weight_sums += weights
            
            # Weighted average
            ensemble[sample_idx, class_idx] = np.sum(preds * weights)
    
    # Report average weight per model
    avg_weights = model_weight_sums / total_decisions
    print("\nAverage weights per model across all decisions:")
    for model_idx, avg_weight in enumerate(avg_weights):
        print(f"  Model {model_idx + 1}: {avg_weight:.4f} ({avg_weight*100:.1f}%)")
    print(f"  (Equal weight would be: {1.0/num_models:.4f} = {100.0/num_models:.1f}%)")
    
    return ensemble
 
 
 
def mean_ensemble(predictions_list: List[np.ndarray]) -> np.ndarray:

    print("\nApplying mean ensemble (equal weights)...")
    return np.mean(predictions_list, axis=0)


def rank_ensemble(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Average rank-normalised predictions. Robust to calibration differences.
    """
    from scipy.stats import rankdata
    
    predictions_array = np.array(predictions_list)  # [M, N, C]
    num_models, num_samples, num_classes = predictions_array.shape
    
    ranked = np.zeros_like(predictions_array)
    for m in range(num_models):
        for c in range(num_classes):
            # Rank then normalise to [0, 1]
            ranks = rankdata(predictions_array[m, :, c])
            ranked[m, :, c] = ranks / num_samples
    
    return np.mean(ranked, axis=0)

def agreement_ensemble(predictions_list):
    
    predictions_array = np.array(predictions_list)
    num_models, num_samples, num_classes = predictions_array.shape
    

    consensus = np.mean(predictions_list, axis=0)
    for _ in range(3):
        deviations = np.mean(np.abs(predictions_list - consensus), axis=(1, 2))
        weights = 1.0 / (deviations + 1e-6)
        weights /= weights.sum()
        consensus = np.sum(predictions_list * weights[:, None, None], axis=0)
    
    return consensus

def load_any_state_dict(path, device='cpu'):
    try:
        import numpy._core.multiarray
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    except (AttributeError, ImportError):
        pass  # Older PyTorch or numpy versions
    
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def extract_text_config(state: Dict) -> Dict:
    """
    Extract text-encoder hyperparameters from a CLIP checkpoint state dict.
    Handles multiple key naming conventions (bare keys vs prefixed with 'text.').
    Only called for non-BioBERT architectures (vit, resnet, swin).
    """
    # token_embedding
    tok_emb = None
    if "token_embedding.weight" in state:
        tok_emb = state["token_embedding.weight"]
    elif "text.token_embedding.weight" in state:
        tok_emb = state["text.token_embedding.weight"]
    else:
        candidates = [v for k, v in state.items() if k.endswith("token_embedding.weight")]
        if candidates:
            tok_emb = candidates[0]

    if tok_emb is None:
        raise KeyError(
            "Cannot find 'token_embedding.weight' in checkpoint. "
            f"First 20 keys: {list(state.keys())[:20]}"
        )

    vocab_size = tok_emb.shape[0]
    tx_width = tok_emb.shape[1]

    # positional_embedding (context length)
    if "positional_embedding" in state:
        ctx_len = state["positional_embedding"].shape[0]
    elif "text.positional_embedding" in state:
        ctx_len = state["text.positional_embedding"].shape[0]
    else:
        ctx_len = 77

    # text transformer layers
    text_layer_keys = [k for k in state if k.startswith("transformer.resblocks.")]
    if not text_layer_keys:
        text_layer_keys = [k for k in state if k.startswith("text.transformer.resblocks.")]
    tx_layers = max([int(k.split('.')[2]) for k in text_layer_keys]) + 1 if text_layer_keys else 12

    # embed_dim from text_projection
    if "text_projection" in state:
        embed_dim = state["text_projection"].shape[1]
    elif "text.text_projection" in state:
        embed_dim = state["text.text_projection"].shape[1]
    else:
        embed_dim = tx_width  # fallback

    return {
        'embed_dim': embed_dim,
        'vocab_size': vocab_size,
        'tx_width': tx_width,
        'tx_heads': tx_width // 64,
        'tx_layers': tx_layers,
        'ctx_len': ctx_len,
    }





def build_model_from_hint(architecture_hint: str, cfg: Dict):
    """
    Build a CLIP model using the architecture hint from the checkpoint directory.

    Args:
        architecture_hint: One of 'vit', 'resnet', 'swin', 'swin_biobert'
        cfg: Text-encoder config from extract_text_config()
    """
    from swin_model import CLIP

    use_biobert = (architecture_hint == 'swin_biobert')

    if architecture_hint == 'vit':
        # Recover ViT vision params from state dict (stored in cfg by caller)
        model = CLIP(
            embed_dim=cfg['embed_dim'],
            image_resolution=cfg['image_resolution'],
            vision_layers=cfg['vision_layers'],
            vision_width=cfg['vision_width'],
            vision_patch_size=cfg['vision_patch_size'],
            context_length=cfg['ctx_len'],
            vocab_size=cfg['vocab_size'],
            transformer_width=cfg['tx_width'],
            transformer_heads=cfg['tx_heads'],
            transformer_layers=cfg['tx_layers'],
            swin_encoder=False,
            use_biobert=False,
        )

    elif architecture_hint == 'resnet':
        model = CLIP(
            embed_dim=cfg['embed_dim'],
            image_resolution=224,
            vision_layers=(3, 4, 6, 3),  # ResNet-50
            vision_width=64,
            vision_patch_size=None,
            context_length=cfg['ctx_len'],
            vocab_size=cfg['vocab_size'],
            transformer_width=cfg['tx_width'],
            transformer_heads=cfg['tx_heads'],
            transformer_layers=cfg['tx_layers'],
            swin_encoder=False,
            use_biobert=False,
        )

    elif architecture_hint in ('swin', 'swin_biobert'):
        model = CLIP(
            embed_dim=cfg['embed_dim'],
            image_resolution=320,   # Placeholder – unused by Swin encoder
            vision_layers=12,       # Placeholder
            vision_width=768,       # Placeholder
            vision_patch_size=16,   # Placeholder
            context_length=cfg['ctx_len'],
            vocab_size=cfg['vocab_size'],
            transformer_width=cfg['tx_width'],
            transformer_heads=cfg['tx_heads'],
            transformer_layers=cfg['tx_layers'],
            swin_encoder=True,
            use_biobert=use_biobert,
            biobert_model='emilyalsentzer/Bio_ClinicalBERT' if use_biobert else None,
        )

    else:
        raise ValueError(f"Unknown architecture hint: '{architecture_hint}'. "
                         "Expected one of: vit, resnet, swin, swin_biobert")

    return model


def build_flava_model(architecture_hint: str, use_biobert: bool = False):
    """
    Build FLAVA model from train.py's load_clip function.
    
    Args:
        architecture_hint: 'flava' or 'flava_biobert'
        use_biobert: Whether to use BioClinicalBERT text encoder
    
    Returns:
        FLAVA model instance
    """
    from train import load_clip
    
    model = load_clip(
        model_path=None,
        pretrained=False,
        context_length=128,
        flava=True,
        use_biobert=use_biobert,
        biobert_model='emilyalsentzer/Bio_ClinicalBERT' if use_biobert else None
    )
    
    return model


def get_eval_transform(model, architecture_hint: str):
    """
    Return the appropriate preprocessing transform for the given architecture.
    """
    if architecture_hint in ('swin', 'swin_biobert'):
        if HAS_TIMM:
            backbone = model.visual if hasattr(model, 'visual') else model
            timm_cfg = resolve_data_config({}, model=backbone)

            H, W = timm_cfg["input_size"][1], timm_cfg["input_size"][2]
            crop_pct = timm_cfg.get("crop_pct", 1.0)
            interp_str = (timm_cfg.get("interpolation", "bicubic") or "bicubic").lower()
            interp_map = {
                "bicubic": InterpolationMode.BICUBIC,
                "bilinear": InterpolationMode.BILINEAR,
                "nearest": InterpolationMode.NEAREST,
            }
            interp = interp_map.get(interp_str, InterpolationMode.BICUBIC)

            if crop_pct and crop_pct < 1.0:
                resize_h = int(math.floor(H / crop_pct))
                resize_w = int(math.floor(W / crop_pct))
            else:
                resize_h, resize_w = H, W

            return Compose([
                Lambda(lambda t: t / 255.0),
                Resize((resize_h, resize_w), interpolation=interp),
                CenterCrop((H, W)),
                Normalize(mean=timm_cfg["mean"], std=timm_cfg["std"]),
            ])
        else:
            return Compose([
                Lambda(lambda t: t / 255.0),
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    elif architecture_hint == 'vit':
        img_res = model.image_resolution if hasattr(model, 'image_resolution') else 224
        return Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(img_res, interpolation=InterpolationMode.BICUBIC),
        ])

    elif architecture_hint == 'resnet':
        return Compose([
            Lambda(lambda t: t / 255.0),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    else:
        raise ValueError(f"Unknown architecture hint: '{architecture_hint}'")


class GPUAdapter(nn.Module):
    """
    GPU adapter that handles CLIP, BioBERT, and FLAVA tokenization.
    Returns CPU tensors for compatibility with existing evaluation code.
    """
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.base = base_model.to(device).eval()
        self.device = device

        # Copy model-specific attributes
        for attr in ('use_biobert', 'use_flava', 'tokenizer',
                    'context_length', 'processor'):
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.encode_image(x.to(self.device, non_blocking=True))
        return out.detach().cpu()

    @torch.no_grad()
    def encode_text(self, t):
        if isinstance(t, dict):
            t = {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
        else:
            t = t.to(self.device, non_blocking=True)
        out = self.base.encode_text(t)
        return out.detach().cpu()

    def __getattr__(self, name):
        if name in {"base", "device", "encode_image", "encode_text",
                    "use_biobert", "use_flava", "tokenizer", "context_length", "processor"}:
            return super().__getattr__(name)
        return getattr(self.base, name)



def tokenize_text(texts, model, context_length=77):
    """
    Tokenize text for CLIP, BioBERT, and FLAVA models.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Check for FLAVA
    if hasattr(model, 'use_flava') and model.use_flava:
        device = next(model.parameters()).device
        tokens = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=model.context_length if hasattr(model, 'context_length') else 128,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in tokens.items()}
    
    # Check for BioBERT (non-FLAVA)
    elif hasattr(model, 'use_biobert') and model.use_biobert:
        device = next(model.parameters()).device
        tokens = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in tokens.items()}
    
    # Standard CLIP
    else:
        import clip
        return clip.tokenize(texts, context_length=context_length)
    

def preprocess_image_flava(images, model):
    """
    Preprocess images using FLAVA's processor.
    
    Args:
        images: Batch of [0,1] scaled tensors Bx3x224x224
        model: FLAVA model instance
        
    Returns:
        pixel_values tensor on correct device
    """
    device = next(model.parameters()).device
    
    # Convert to PIL images for FLAVA processor
    pil_images = [to_pil_image(img.cpu()) for img in images]
    
    processed = model.processor(
        images=pil_images,
        return_tensors="pt"
    )
    return processed['pixel_values'].to(device)

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple,
                     context_length: int = 77, is_flava: bool = False):
    """
    Run softmax evaluation with proper tokenization for CLIP, BioBERT, and FLAVA.
    """
    from tqdm import tqdm
    
    pos, neg = pair_template

    def get_weights(template):
        weights = []
        for classname in tqdm(eval_labels):
            texts = [template.format(classname)]
            tokens = tokenize_text(texts, model, context_length)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            emb = emb / emb.norm()
            weights.append(emb)
        return torch.stack(weights, dim=1)

    zeroshot_weights_pos = get_weights(pos)
    zeroshot_weights_neg = get_weights(neg)

    pos_pred = predict(loader, model, zeroshot_weights_pos, is_flava=is_flava)
    neg_pred = predict(loader, model, zeroshot_weights_neg, is_flava=is_flava)

    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred

    return y_pred


def predict(loader, model, zeroshot_weights, is_flava: bool = False):
    """
    Prediction loop compatible with CLIP, BioBERT, and FLAVA.
    """
    from tqdm import tqdm

    y_pred = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Running inference"):
            images = data['img']
            
            # FLAVA uses processor, CLIP uses direct encoding
            if is_flava:
                pixel_values = preprocess_image_flava(images, model)
                image_features = model.encode_image(pixel_values)
            else:
                image_features = model.encode_image(images)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ zeroshot_weights
            logits = np.squeeze(logits.cpu().numpy(), axis=0)
            y_pred.append(logits)

    return np.array(y_pred)


def make_model_from_checkpoint(model_path: str, cxr_h5: str, architecture_hint: str):
    """
    Load model from checkpoint using the architecture hint from the directory name.

    Args:
        model_path: Path to checkpoint file
        cxr_h5: Path to HDF5 image file
        architecture_hint: Architecture type

    Returns:
        Tuple of (model, dataloader, is_flava)
    """
    print(f"\nLoading: {Path(model_path).name}")

    arch_display = architecture_hint.upper().replace('_', ' + ').replace('BIOBERT', 'BioBERT')
    print(f"  Architecture: {arch_display}")

    is_flava = architecture_hint in ('flava', 'flava_biobert')
    
    # FLAVA models

    if is_flava:
        use_biobert = (architecture_hint == 'flava_biobert')
        
        # Build FLAVA model
        base = build_flava_model(architecture_hint, use_biobert=use_biobert)
        
        # Load weights
        state = load_any_state_dict(model_path, device='cpu')
        result = base.load_state_dict(state, strict=False)
        
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        
        base.eval()
        
        # FLAVA uses its own dataset
        if USE_GPU:
            device = torch.device("cuda")
            model = GPUAdapter(base, device)
            pin = True
            print("  Using GPU")
        else:
            model = base
            pin = False
            print("  Using CPU")
        
        dset = CXRTestDatasetFLAVA(img_path=cxr_h5)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )
        
        return model, loader, is_flava
    

    # swin_biobert: instantiate CLIP directly from swin_model.py

    elif architecture_hint == 'swin_biobert':
        from swin_model import CLIP

        state = load_any_state_dict(model_path, device='cpu')

        # Infer embed_dim from the projection head in the checkpoint
        embed_dim = state["visual.head.weight"].shape[0]
        print(f"  Embed dim: {embed_dim} (from checkpoint)")

        model = CLIP(
            embed_dim=embed_dim,
            image_resolution=320,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            swin_encoder=True,
            use_biobert=True,
            biobert_model='emilyalsentzer/Bio_ClinicalBERT',
        )
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        model.eval()
        transform = get_eval_transform(model, architecture_hint)


    # All other CLIP architectures (vit, resnet, swin)

    else:
        state = load_any_state_dict(model_path, device='cpu')
        cfg = extract_text_config(state)

        if cfg is None:
            raise KeyError(
                f"No CLIP text encoder keys found in checkpoint for architecture "
                f"'{architecture_hint}'. First 20 keys: {list(state.keys())[:20]}"
            )

        # ViT also needs vision params from the state dict
        if architecture_hint == 'vit':
            conv1_key = next((k for k in state if k.endswith("visual.conv1.weight")), None)
            pos_key   = next((k for k in state if k.endswith("visual.positional_embedding")), None)
            if conv1_key is None or pos_key is None:
                raise KeyError(
                    "Cannot find ViT vision keys in checkpoint. "
                    f"First 20 keys: {list(state.keys())[:20]}"
                )
            vision_width  = state[conv1_key].shape[0]
            patch_size    = state[conv1_key].shape[2]
            n_plus1       = state[pos_key].shape[0]
            grid          = int(round((n_plus1 - 1) ** 0.5))
            prefix        = conv1_key[: conv1_key.index("visual.conv1.weight")]
            vision_layer_keys = [k for k in state if k.startswith(f"{prefix}visual.transformer.resblocks.")]
            vision_layers = max([int(k.split('.')[3]) for k in vision_layer_keys]) + 1
            cfg.update({
                'image_resolution': grid * patch_size,
                'vision_layers': vision_layers,
                'vision_width': vision_width,
                'vision_patch_size': patch_size,
            })

        print(f"  Embed dim: {cfg['embed_dim']}")
        print(f"  Context length: {cfg['ctx_len']}")

        model = build_model_from_hint(architecture_hint, cfg)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        model.eval()
        transform = get_eval_transform(model, architecture_hint)

    # Wrap with GPU and create loader (for non-FLAVA models)
    if not is_flava:
        if USE_GPU:
            device = torch.device("cuda")
            model = GPUAdapter(model, device)
            pin = True
            print("  Using GPU")
        else:
            pin = False
            print("  Using CPU")

        dset = CXRTestDataset(cxr_h5, transform=transform)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )

    return model, loader, is_flava


def ensemble_models(
    architecture_checkpoints: Dict[str, List[str]],
    cxr_h5: str,
    cxr_labels: List[str],
    cxr_pair_template: Tuple[str, str],
    cache_dir: Optional[Path] = None,
    save_name: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Ensemble predictions from multiple checkpoints organized by architecture.
    """
    predictions = []

    for arch_name, model_paths in architecture_checkpoints.items():
        if not model_paths:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {arch_name.upper()} models ({len(model_paths)} checkpoint(s))")
        print(f"{'='*60}")

        for path in model_paths:
            model_name = Path(path).stem

            cache_path = None
            if cache_dir is not None:
                cache_dir.mkdir(exist_ok=True, parents=True)
                cache_file = (f"{save_name}_{arch_name}_{model_name}.npy"
                              if save_name else f"{arch_name}_{model_name}.npy")
                cache_path = cache_dir / cache_file

            if cache_path is not None and cache_path.exists():
                print(f"\nLoading cached: {model_name}")
                y_pred = np.load(cache_path)
            else:
                model, loader, is_flava = make_model_from_checkpoint(
                    path, cxr_h5, architecture_hint=arch_name
                )

                with torch.no_grad():
                    context_length = model.context_length if hasattr(model, 'context_length') else 77
                    y_pred = run_softmax_eval(
                        model, loader, cxr_labels, cxr_pair_template,
                        context_length=context_length,
                        is_flava=is_flava
                    )

                if cache_path is not None:
                    np.save(cache_path, y_pred)
                    print(f"  Cached to: {cache_path.name}")

                del model, loader
                torch.cuda.empty_cache()

            predictions.append(y_pred)

    if not predictions:
        raise ValueError("No checkpoints found in any directory!")

    y_pred_avg = np.mean(predictions, axis=0)

    print(f"\n{'='*60}")
    print("Ensemble Summary")
    print(f"{'='*60}")
    print(f"Total models: {len(predictions)}")
    for arch_name, paths in architecture_checkpoints.items():
        if paths:
            print(f"  {arch_name}: {len(paths)} model(s)")

    return predictions, y_pred_avg


# ================= MAIN =================
if __name__ == "__main__":
    print("=" * 60)
    print("Universal Ensemble Evaluation")
    print("CLIP: ViT, ResNet, Swin, Swin+BioBERT")
    print("FLAVA: Standard BERT, BioClinicalBERT")
    print(f"Strategy: {args.ensemble_strategy.upper()}")
    print("=" * 60)
 
    print("\nScanning checkpoint directories...")
    architecture_checkpoints = collect_model_paths_by_architecture(checkpoint_dirs)
 
    total_checkpoints = sum(len(paths) for paths in architecture_checkpoints.values())
 
    if total_checkpoints == 0:
        print("\nNo checkpoints found!")
        sys.exit(1)
 
    print(f"\nTotal: {total_checkpoints} checkpoint(s)")
 
    print("\n" + "=" * 60)
    print("Running Test Set Evaluation")
    print("=" * 60)
 
    # Collect individual predictions
    preds_list, _ = ensemble_models(
        architecture_checkpoints=architecture_checkpoints,
        cxr_h5=cxr_filepath,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        cache_dir=cache_dir,
    )
 
    # Apply ensemble strategy
    print("\n" + "=" * 60)
    print(f"Applying Ensemble Strategy: {args.ensemble_strategy.upper()}")
    print("=" * 60)
 
    if args.ensemble_strategy == 'uncertainty':
        y_pred_ensemble = uncertainty_weighted_ensemble(preds_list)
    elif args.ensemble_strategy == 'rank':
        y_pred_ensemble = rank_ensemble(preds_list)
    elif args.ensemble_strategy == 'agreement':
        y_pred_ensemble = agreement_ensemble(preds_list)
    else:
        y_pred_ensemble = mean_ensemble(preds_list)
 

    pred_path = predictions_dir / f"chexpert_preds_{args.ensemble_strategy}.npy"
    np.save(pred_path, y_pred_ensemble)
    print(f"\nSaved ensemble predictions to {pred_path}")
 
    # Compute AUC
    print("\n" + "=" * 60)
    print("Computing AUC Metrics")
    print("=" * 60)
 
    test_true = make_true_labels(cxr_true_labels_path, cxr_labels)
    auc_df = evaluate(y_pred_ensemble, test_true, cxr_labels)
    print("\nPer-label AUCs:")
    print(auc_df)
 
    mean_auc = auc_df.mean(axis=1)[0]
    print(f"\nMean AUC: {mean_auc:.4f}")
 
    # Bootstrap CI
    print("\n" + "=" * 60)
    print("Computing Bootstrap Confidence Intervals")
    print("=" * 60)
 
    auc_ci_df = bootstrap(y_pred_ensemble, test_true, cxr_labels)[1]
    print("\nAUC with 95% CI:")
    print(auc_ci_df)
 
    auc_ci_csv = predictions_dir / f"chexpert_auc_ci_{args.ensemble_strategy}.csv"
    auc_ci_df.to_csv(auc_ci_csv, index=True)
    print(f"\nSaved AUC with CI to: {auc_ci_csv}")
 
    # F1 & MCC with validation thresholds
    if os.path.exists(val_img_path) and os.path.exists(val_label_path):
        print("\n" + "=" * 60)
        print("Computing F1 and MCC with Validation Thresholds")
        print("=" * 60)
 
        try:
            val_true = make_true_labels(val_label_path, cxr_labels)
            val_preds_list, _ = ensemble_models(  # ← Fixed: get list, not average
                architecture_checkpoints=architecture_checkpoints,
                cxr_h5=val_img_path,
                cxr_labels=cxr_labels,
                cxr_pair_template=cxr_pair_template,
                cache_dir=cache_dir,
                save_name="val",
            )
 
            # Apply same ensemble strategy to validation
            if args.ensemble_strategy == 'uncertainty':
                val_pred_ensemble = uncertainty_weighted_ensemble(val_preds_list)
            elif args.ensemble_strategy == 'rank':
                val_pred_ensemble = rank_ensemble(val_preds_list)
            elif args.ensemble_strategy == 'agreement':
                val_pred_ensemble = agreement_ensemble(val_preds_list)
            else:
                val_pred_ensemble = mean_ensemble(val_preds_list)
 
            best_p_vals = get_best_p_vals(val_pred_ensemble, val_true, cxr_labels)
 
            f1_ci = compute_f1(y_pred_ensemble, test_true, cxr_labels, best_p_vals)
            mcc_ci = compute_mcc(y_pred_ensemble, test_true, cxr_labels, best_p_vals)

            f1_csv = predictions_dir / f"chexpert_f1_ci_{args.ensemble_strategy}.csv"
            mcc_csv = predictions_dir / f"chexpert_mcc_ci_{args.ensemble_strategy}.csv"
            f1_ci.to_csv(f1_csv, index=True)
            mcc_ci.to_csv(mcc_csv, index=True)

            print("\nF1 Scores with 95% CI:")
            print(f1_ci)
            print("\nMCC Scores with 95% CI:")
            print(mcc_ci)

            print(f"\nSaved F1 with CI to: {f1_csv}")
            print(f"\nSaved MCC with CI to: {mcc_csv}")

        except Exception as e:
            print(f"\nCould not compute F1/MCC: {e}")
    else:
        print("\nSkipping F1/MCC: validation data not found")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {predictions_dir}")
    print(f"  - Predictions: {pred_path.name}")
    print(f"  - AUC with CI: {auc_ci_csv.name}")

    print("\nModels used:")
    for arch_name, paths in architecture_checkpoints.items():
        if paths:
            print(f"  {arch_name}: {len(paths)} model(s)")
            for path in paths:
                print(f"    - {Path(path).name}")