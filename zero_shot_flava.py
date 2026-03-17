"""
zero_shot_flava.py - Zero shot evaluation for FLAVA models
Supports both standard BERT and BioClinicalBERT text encoders
"""
import numpy as np
import os
import sys
import pandas as pd
import h5py
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils import data
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, InterpolationMode
from torchvision.transforms.functional import to_pil_image

from eval import evaluate, sigmoid, bootstrap


class CXRTestDatasetFLAVA(data.Dataset):
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


def load_test_data_flava(cxr_path: str):
    """
    Load test data for FLAVA evaluation.

    Args:
        cxr_path: Path to HDF5 file containing CXR images

    Returns:
        DataLoader with FLAVA-appropriate preprocessing
    """
    dataset = CXRTestDatasetFLAVA(img_path=cxr_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return loader


def tokenize_text_flava(texts, model):
    """
    Tokenize text for FLAVA models, handling both
    standard BERT and BioClinicalBERT tokenizers.

    Args:
        texts: String or list of strings
        model: FLAVAWrapper model instance

    Returns:
        Dict of tokenized inputs on correct device
    """
    if isinstance(texts, str):
        texts = [texts]

    device = next(model.parameters()).device

    tokens = model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=model.context_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in tokens.items()}


def preprocess_image_flava(images, model):
    """
    Preprocess images using FLAVA's processor.

    Args:
        images: Batch of [0,1] scaled tensors Bx3x224x224
        model: FLAVAWrapper model instance

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


def zeroshot_classifier_flava(classnames, templates, model):
    """
    Generate zero-shot classifier weights for FLAVA models.

    Args:
        classnames: List of class names e.g. ['Atelectasis', ...]
        templates: List of template strings with {} placeholder
        model: FLAVAWrapper model instance

    Returns:
        Tensor of shape [embed_dim, num_classes]
    """
    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(classnames):
            # Format templates with class name
            texts = [template.format(classname) for template in templates]

            # Tokenize using FLAVA-appropriate tokenizer
            text_tokens = tokenize_text_flava(texts, model)

            # Encode text through FLAVA text encoder
            class_embeddings = model.encode_text(text_tokens)

            # Normalize
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # Average over templates
            class_embedding = class_embeddings.mean(dim=0)

            # Normalize averaged embedding
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights


def predict_flava(loader, model, zeroshot_weights, softmax_eval=True):
    """
    Run predictions on data loader using FLAVA model.

    Args:
        loader: PyTorch DataLoader (CXRTestDatasetFLAVA)
        model: FLAVAWrapper model instance
        zeroshot_weights: Pre-computed text embeddings [embed_dim, num_classes]
        softmax_eval: Whether to use softmax evaluation

    Returns:
        Numpy array of predictions [num_samples, num_classes]
    """
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img']  # [0,1] scaled Bx3x224x224

            # Preprocess through FLAVA processor
            pixel_values = preprocess_image_flava(images, model)

            # Encode image
            image_features = model.encode_image(pixel_values)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logits = image_features @ zeroshot_weights
            logits = np.squeeze(logits.cpu().numpy(), axis=0)

            if not softmax_eval:
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits)

            y_pred.append(logits)

    return np.array(y_pred)


def run_single_prediction_flava(cxr_labels, template, model, loader,
                                 softmax_eval=True):
    """
    Make predictions for a single template using FLAVA.

    Args:
        cxr_labels: List of class labels
        template: Single template string e.g. '{}'
        model: FLAVAWrapper model instance
        loader: Data loader
        softmax_eval: Use softmax evaluation

    Returns:
        Predictions array [num_samples, num_classes]
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier_flava(
        cxr_labels, cxr_phrase, model
    )
    y_pred = predict_flava(
        loader, model, zeroshot_weights, softmax_eval=softmax_eval
    )
    return y_pred


def run_softmax_eval_flava(model, loader, eval_labels: list,
                            pair_template: tuple):
    """
    Run softmax evaluation for FLAVA model using
    positive/negative template pairs.

    Args:
        model: FLAVAWrapper model instance
        loader: Data loader (CXRTestDatasetFLAVA)
        eval_labels: List of pathology labels
        pair_template: Tuple of (positive_template, negative_template)
                       e.g. ('{}', 'no {}')

    Returns:
        Predictions array [num_samples, num_classes]
    """
    pos = pair_template[0]
    neg = pair_template[1]

    pos_pred = run_single_prediction_flava(
        eval_labels, pos, model, loader, softmax_eval=True
    )
    neg_pred = run_single_prediction_flava(
        eval_labels, neg, model, loader, softmax_eval=True
    )

    # Softmax over positive and negative predictions
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred

    return y_pred


def make_true_labels(cxr_true_labels_path: str,
                     cxr_labels: List[str],
                     cutlabels: bool = True):
    """
    Load ground truth labels from CSV.

    Args:
        cxr_true_labels_path: Path to CSV with ground truth
        cxr_labels: List of label column names to extract
        cutlabels: If True, select only specified columns

    Returns:
        Numpy array [num_samples, num_labels]
    """
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)
    return full_labels.to_numpy()