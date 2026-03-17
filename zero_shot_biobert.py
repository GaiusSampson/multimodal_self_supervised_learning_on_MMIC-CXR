"""
Modified zero_shot.py with BioBERT support
"""
import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import clip
from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

# Import tokenization helpers
try:
    from simple_tokenizer import SimpleTokenizer
except:
    from train import preprocess_text


class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset for testing."""
    def __init__(self, img_path: str, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx]  # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
        return sample


def tokenize_text(texts, model, context_length=77):
    """
    Tokenize text for either standard CLIP or BioBERT models.
    
    Args:
        texts: List of strings or single string
        model: CLIP model (either standard or with BioBERT)
        context_length: Max tokens for standard CLIP
    
    Returns:
        Tokenized text in appropriate format
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Check if model uses BioBERT
    if hasattr(model, 'use_biobert') and model.use_biobert:
        # Use BERT tokenizer
        device = next(model.parameters()).device
        tokens = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,  # BERT max length
            return_tensors="pt"
        )
        # Move to device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens
    else:
        # Use standard CLIP tokenizer
        return clip.tokenize(texts, context_length=context_length)


def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    Generate zero-shot classifier weights for given classes and templates.
    Works with both standard CLIP and BioBERT text encoders.
    
    Args:
        classnames: List of class names
        templates: List of template strings with {} placeholder
        model: CLIP model
        context_length: Max tokens (for standard CLIP)
    
    Returns:
        Tensor of shape [embed_dim, num_classes]
    """
    with torch.no_grad():
        zeroshot_weights = []
        
        for classname in tqdm(classnames):
            # Format templates with class name
            texts = [template.format(classname) for template in templates]
            
            # Tokenize (handles both CLIP and BioBERT)
            text_tokens = tokenize_text(texts, model, context_length)
            
            # Encode text
            class_embeddings = model.encode_text(text_tokens)
            
            # Normalize embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            # Average over templates
            class_embedding = class_embeddings.mean(dim=0)
            
            # Normalize again
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    
    return zeroshot_weights


def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0): 
    """
    Run predictions on data loader.
    
    Args:
        loader: PyTorch DataLoader
        model: CLIP model
        zeroshot_weights: Pre-computed text embeddings
        softmax_eval: Whether to use softmax evaluation
        verbose: Print debug info
    
    Returns:
        Numpy array of predictions [num_samples, num_classes]
    """
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img']

            # Encode image
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logits = image_features @ zeroshot_weights
            logits = np.squeeze(logits.cpu().numpy(), axis=0)
        
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits)
            
            y_pred.append(logits)
            
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)


def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77): 
    """
    Make predictions for a single template.
    
    Args:
        cxr_labels: List of class labels
        template: Single template string
        model: CLIP model
        loader: Data loader
        softmax_eval: Use softmax evaluation
        context_length: Max tokens for CLIP
    
    Returns:
        Predictions array
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred


def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77, flava=None): 
    """
    Run softmax evaluation with positive/negative template pairs.
    
    Args:
        model: CLIP model
        loader: Data loader
        eval_labels: List of labels
        pair_template: Tuple of (positive_template, negative_template)
        context_length: Max tokens
        flava: Legacy parameter (kept for compatibility)
    
    Returns:
        Predictions using softmax over positive/negative pairs
    """
    # Get positive and negative phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # Get predictions for both
    pos_pred = run_single_prediction(
        eval_labels, pos, model, loader,
        softmax_eval=True, context_length=context_length
    )
    neg_pred = run_single_prediction(
        eval_labels, neg, model, loader,
        softmax_eval=True, context_length=context_length
    )

    # Compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    
    return y_pred


def make_true_labels(cxr_true_labels_path: str, cxr_labels: List[str], cutlabels: bool = True): 
    """
    Load ground truth labels.
    
    Args:
        cxr_true_labels_path: Path to CSV with labels
        cxr_labels: List of label columns to extract
        cutlabels: Whether to select specific columns
    
    Returns:
        Numpy array of shape [num_samples, num_labels]
    """
    full_labels = pd.read_csv(cxr_true_labels_path)
    
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true