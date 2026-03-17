import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval
from metrics import compute_f1_mcc

## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = './test_data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = './test_data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = './checkpointsViT/pt-imp' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('./predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

X_val_dir = "./test_data/chexpert_val.h5"
y_val_dir = "./test_data/val.csv"

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)
        
print(model_paths)

def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
)

# save averaged preds
pred_name = "chexpert_preds.npy" # add name of preds
predictions_dir = predictions_dir / pred_name
np.save(file=predictions_dir, arr=y_pred_avg)

# make test_true
test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# Ensemble predictions for validation set to get optimal thresholds
val_predictions, val_y_pred_avg = ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=X_val_dir, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
    save_name="val"
)

print(f"Validation predictions shape: {val_y_pred_avg.shape}")
print(f"Contains NaN: {np.isnan(val_y_pred_avg).any()}")
print(f"NaN count: {np.isnan(val_y_pred_avg).sum()}")
print(f"Min value: {np.nanmin(val_y_pred_avg)}")
print(f"Max value: {np.nanmax(val_y_pred_avg)}")

print("\n=== Checking individual model predictions ===")
for i, (path, pred) in enumerate(zip(model_paths, val_predictions)):
    model_name = Path(path).stem
    nan_count = np.isnan(pred).sum()
    total = pred.size
    print(f"{model_name}: {nan_count}/{total} NaNs ({100*nan_count/total:.1f}%)")
    if nan_count == 0:
        print(f"  Range: [{pred.min():.4f}, {pred.max():.4f}]")

# Check which labels/samples have NaN
if np.isnan(val_y_pred_avg).any():
    nan_locations = np.argwhere(np.isnan(val_y_pred_avg))
    print(f"NaN locations (sample_idx, label_idx):\n{nan_locations[:10]}") 

# Load validation ground truth
val_true = make_true_labels(cxr_true_labels_path=y_val_dir, cxr_labels=cxr_labels)

# Get best thresholds using ensembled validation predictions
from metrics import get_best_p_vals, compute_f1, compute_mcc

best_p_vals = get_best_p_vals(val_y_pred_avg, val_true, cxr_labels)

# Compute F1 and MCC using ensembled test predictions (already computed as y_pred_avg)
print("\nComputing F1 and MCC scores with 95% CI...")
f1_cis = compute_f1(test_pred, test_true, cxr_labels, best_p_vals)
mcc_cis = compute_mcc(test_pred, test_true, cxr_labels, best_p_vals)


print("\nF1 Scores [mean, lower, upper]:")
print(f1_cis)
print("\nMCC Scores [mean, lower, upper]:")
print(mcc_cis)

# evaluate model
cxr_results = evaluate(test_pred, test_true, cxr_labels)

# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)

# display AUC with confidence intervals
bootstrap_results[1]

print("AUC results ", bootstrap_results[1])
        
# copy the outpts to a text file so they can be read more easily

f1_cis.to_csv(os.path.join(cache_dir, "f1_scores.csv"))
mcc_cis.to_csv(os.path.join(cache_dir, "mcc_scores.csv"))
bootstrap_results[1].to_csv(os.path.join(cache_dir, "auc_scores.csv"))
