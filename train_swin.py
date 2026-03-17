import os
import argparse
import logging
import warnings
from datetime import datetime
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from train import load_data, load_clip, preprocess_text
from zero_shot import make_true_labels, run_softmax_eval
from eval import evaluate
import numpy as np

# =========================
# Logging & Warnings
# =========================
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_swin_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    # Secondary handler for warnings file
    logging.basicConfig(
        filename='warnings.log',
        filemode='w',
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{category.__name__}: {message} (from {filename}:{lineno})")

    warnings.showwarning = custom_warning_handler
    warnings.filterwarnings("once", message="`torch.utils._pytree._register_pytree_node` is deprecated")
    warnings.filterwarnings("once", message="Importing from timm.models.layers is deprecated")
    warnings.filterwarnings("once", message="`resume_download` is deprecated")
    warnings.filterwarnings("once", message="You are using `torch.load` with `weights_only=False`")

from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from zero_shot import CXRTestDataset

def load_test_data(cxr_path, pretrained=True):
    transformations = [
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    if pretrained:
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    torch_dset = CXRTestDataset(img_path=cxr_path, transform=transform)
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    return loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=4)  # total epochs to run
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--base_model', type=str, default="SWINV2", help="The pretrained CLIP base model")
    parser.add_argument('--use_biobert', action='store_true',
                        help='Use BioClinicalBERT for text encoder')
    parser.add_argument('--biobert_model', type=str,
                        default='emilyalsentzer/Bio_ClinicalBERT',
                        choices=[
                            'emilyalsentzer/Bio_ClinicalBERT',
                            'emilyalsentzer/Bio_Discharge_Summary_BERT',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                        ],
                        help='BioBERT model variant to use')
    parser.add_argument('--flava', help="Finetune on FLAVA instead on CLIP", action='store_true')

    # ---- resume options ----
    parser.add_argument('--resume_path', type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument('--resume_auto', action='store_true', help="Auto-load checkpoints/<model_name>/last.pt if found")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    setup_logging()
    if config.run_dir is None:
        config.run_dir = f"./runs/{config.model_name}"
    
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer, minimizer = make(config)

    # resume (optional)
    start_epoch = load_checkpoint_if_any(
        model, optimizer, device,
        save_dir=config.save_dir,
        model_name=config.model_name,
        resume_path=config.resume_path,
        resume_auto=config.resume_auto
    )

    # train
    train(model, data_loader, device, criterion, optimizer, config, minimizer, start_epoch=start_epoch)

    # final save
    final_path = os.path.join(config.save_dir, str(config.model_name), 'final.pt')
    save_checkpoint(model, optimizer, config.epochs, final_path)
    logging.info(f"Saved final checkpoint to {final_path}")

    if verbose: 
        logging.info(str(model))
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(
        config.cxr_filepath, 
        config.txt_filepath, 
        batch_size=config.batch_size, 
        pretrained=pretrained, 
        column="impression"
    )
    
    if config.base_model != "SWINV2":
        model = load_clip(
            model_path=None, 
            pretrained=pretrained, 
            context_length=config.context_length, 
            pretrained_model=config.base_model
        )
    else:
        model = load_clip(
                model_path=None,
                pretrained=False,
                context_length=args.context_length,
                swin_encoder=True,
                use_biobert=args.use_biobert,
                biobert_model=args.biobert_model
            )
    model = model.float()
    model.to(device)
    model.train()
    logging.info('Model on Device.')
    
    parameters = model.parameters()
    
    # optimizer
    minimizer = None
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(parameters, lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(parameters, lr=config.lr, momentum=config.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
     
    return model, data_loader, device, criterion, optimizer, minimizer

def accuracy(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    correct_preds = np.sum(y_true == y_pred_binary)
    total_preds = y_true.size
    return correct_preds / total_preds

def train(model, loader, device, criterion, optimizer, config, minimizer=None, start_epoch=0): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        os.makedirs(model_save_dir)
    
    example_ct = 0
    batch_ct = 0
    report_freq = config.log_interval
    
    # validation setup
    cxr_labels = ['Atelectasis','Cardiomegaly',
                  'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                  'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                  'Pneumothorax', 'Support Devices']
    cxr_pair_template = ("{}", "no {}")
    test_loader = load_test_data("test_data/chexpert_val.h5")
    y_true = make_true_labels("test_data/val.csv", cxr_labels)
    
    # track top-10 by AUC
    best_models = {i:{"Filepath": os.path.join(model_save_dir, f"Best_{i}.pt"), "Mean_AUC": 0} for i in range(10)}
  
    for epoch in range(start_epoch, config.epochs):
        running_loss = 0.0
        logging.info(f"Starting epoch {epoch+1}/{config.epochs}")

        for data in tqdm(loader):
            images = data['img']
            texts = data['txt']
            
            if not config.flava:
                texts = preprocess_text(texts, model) 
                
            loss = train_batch(config, images, texts, model, device, criterion, optimizer, minimizer)
            
            example_ct += len(images)
            batch_ct += 1
            running_loss += loss.item()
            
            if (batch_ct % report_freq) == 0:
                avg_loss = running_loss / report_freq
                logging.info(f"[epoch {epoch+1} | batch {batch_ct}] loss={avg_loss:.4f} seen={example_ct}")
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0:
                # Evaluate and possibly save into top-10
                model = model.cpu()
                model.eval()
                y_pred = run_softmax_eval(model, test_loader, cxr_labels, cxr_pair_template, flava=None)
                model = model.to(device)
                stats = evaluate(y_pred, y_true, cxr_labels)
                mean_auc = stats.mean(1)[0]
                model.train()
                
                smallest_score = 101
                smallest_index = 0
                for k, v in best_models.items():
                    if v["Mean_AUC"] < smallest_score:
                        smallest_score = v["Mean_AUC"]
                        smallest_index = k

                if mean_auc > smallest_score:
                    logging.info(f'Validation AUC {mean_auc:.4f} enters top-10 -> {best_models[smallest_index]["Filepath"]}')
                    best_models[smallest_index]["Mean_AUC"] = mean_auc
                    save(model, best_models[smallest_index]["Filepath"])
                    
            del images, texts
            torch.cuda.empty_cache()

        # ======= END OF EPOCH: save checkpoints =======
        epoch_ckpt_path = os.path.join(model_save_dir, f"epoch{epoch+1:03d}.pt")
        save_checkpoint(model, optimizer, epoch+1, epoch_ckpt_path)
        save_checkpoint(model, optimizer, epoch+1, os.path.join(model_save_dir, "last.pt"))  # rolling last
        logging.info(f"[Epoch {epoch+1}] saved epoch checkpoint -> {epoch_ckpt_path}")

def train_batch(config, images, texts, model, device, criterion, optimizer, minimizer=None):
    if not config.flava:
        if model.use_biobert:
        # texts is already a dict on device from preprocess_text
            images = images.to(device)
        else:
            images, texts = images.to(device), texts.to(device)
        batch_size = images.shape[0]
    else:
        images = [img for img in images]
        batch_size = len(images)
        images = model.image_processor(images, return_tensors="pt").to(device)
        texts = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        
    logits_per_image, logits_per_text = model(images, texts)

    labels = torch.arange(batch_size).to(device)
    
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2

    loss.backward()
    
    if minimizer is not None:
        minimizer.ascent_step()
        loss_temp = loss
        logits_per_image, logits_per_text = model(images, texts)
        loss_img = criterion(logits_per_image, labels)
        loss_txt = criterion(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2
        loss.backward()
        minimizer.descent_step()
        loss = loss_temp
    else:
        optimizer.step()
        
    optimizer.zero_grad()
    return loss

def save(model, path): 
    torch.save(model.state_dict(), path)

# Save model + optimizer + epoch (resumable)
def save_checkpoint(model, optimizer, epoch, path, extra=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    torch.save(ckpt, path)

def load_checkpoint_if_any(model, optimizer, device, save_dir, model_name, resume_path="", resume_auto=False):
    """
    Load a checkpoint if provided; return start_epoch (int).
    If resuming from a file saved at epoch N, training will continue from epoch N (next is N+1).
    """
    path = None
    if resume_path:
        path = resume_path
    elif resume_auto:
        candidate = os.path.join(save_dir, model_name, "last.pt")
        if os.path.exists(candidate):
            path = candidate

    if path and os.path.exists(path):
        logging.info(f"Loading checkpoint from: {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        logging.info(f"Resumed at epoch {start_epoch}. Will continue up to --epochs={model_name if False else ''}")
        return start_epoch
    else:
        if path:
            logging.warning(f"Requested resume file not found: {path}")
        return 0

if __name__ == "__main__":
    args = parse_args()
    _ = model_pipeline(args)
