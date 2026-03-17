import os
import argparse
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from train import load_data, load_clip, preprocess_text
from load_test_data import load_test_data
from zero_shot import make_true_labels, run_softmax_eval
from torch.optim.lr_scheduler import CosineAnnealingLR
from eval import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--effective_batch_size', type=int, default=64, help='the effective batch size make this a multiple of the batch size '
    'for best results keep it between 1x and 4x batch size')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4) #5e-6 for rn50
    parser.add_argument('--minimum_lr', type=float, default=1e-9, help='the final learning rate when scheduling, set the same as lr to not use scheduling')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default="checkpointstest/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")#adam for rn50
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer, minimizer, scheduler = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config, minimizer, scheduler)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length, use_biobert=False, biobert_model=None)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    minimizer = None
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    batches_per_epoch = 377112 / config.batch_size
    total_batches = batches_per_epoch * config.epochs // (config.effective_batch_size/config.batch_size) # accumulation steps
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_batches,
        eta_min=config.minimum_lr
    )
     
    return model, data_loader, device, criterion, optimizer, minimizer, scheduler

def train(model, loader, device, criterion, optimizer, config, minimizer, scheduler): 
    """
    Main training loop with gradient accumulation.
    """
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Calculate accumulation steps
    target_effective_batch = config.effective_batch_size
    physical_batch = config.batch_size
    accumulation_steps = max(1, target_effective_batch // physical_batch)
    
    print(f"Physical batch size: {physical_batch}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {physical_batch * accumulation_steps}")

    example_ct = 0
    batch_ct = 0
    running_loss = 0.0
    report_freq = config.log_interval
    
    # Validation setup
    cxr_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    cxr_pair_template = ("{}", "no {}")
    
    # Load validation data

    test_loader = load_test_data("test_data/chexpert_val.h5")
    y_true = make_true_labels("test_data/val.csv", cxr_labels)
    print("Validation data loaded successfully")
    
    # Track top-10 models
    best_models = {
        i: {
            "Filepath": os.path.join(model_save_dir, f"Best_{i}.pt"),
            "Mean_AUC": 0
        } for i in range(10)
    }
    
    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")
        model.train()
        
        # Zero gradients at start of epoch
        optimizer.zero_grad()
        
        for data in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = data['img']
            texts = data['txt']
            
            # Train on batch (accumulates gradients)
            loss = train_batch(
                images, texts, model, device, criterion, optimizer,
                accumulation_steps,
            )
            
            example_ct += len(images)
            batch_ct += 1
            running_loss += loss
            
            # Step optimizer every accumulation_steps batches
            if batch_ct % accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            
            # Log progress
            if (batch_ct % config.log_interval) == 0:
                train_log(running_loss / report_freq, example_ct, batch_ct, epoch)
                running_loss = 0.0
            
            # Validation and checkpointing
            if (batch_ct % config.save_interval) == 0 and test_loader is not None:
                print("Running validation...")
                
                # Make sure gradients are stepped before validation
                if batch_ct % accumulation_steps != 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                
                # Move to CPU for validation
                model_cpu = model.cpu()
                model_cpu.eval()
                
                try:
                    with torch.no_grad():
                        y_pred = run_softmax_eval(
                            model_cpu, test_loader, cxr_labels, 
                            cxr_pair_template, context_length=config.context_length
                        )
                    
                    # Evaluate
                    stats = evaluate(y_pred, y_true, cxr_labels)
                    mean_auc = stats.mean(1)[0]
                    print(f"Validation AUC: {mean_auc:.4f}")
                    
                    # Check if top-10
                    smallest_score = 101
                    smallest_index = 0
                    for k, v in best_models.items():
                        if v["Mean_AUC"] < smallest_score:
                            smallest_score = v["Mean_AUC"]
                            smallest_index = k
                    
                    if mean_auc > smallest_score:
                        print(
                            f'Validation AUC {mean_auc:.4f} in top-10 saving to '
                            f'{best_models[smallest_index]["Filepath"]}'
                        )
                        best_models[smallest_index]["Mean_AUC"] = mean_auc
                        torch.save(model_cpu.state_dict(), 
                                 best_models[smallest_index]["Filepath"])
                    else:
                        print(
                            f'Validation AUC {mean_auc:.4f} not in top-10'
                        )
                
                except Exception as e:
                    print(f"Validation failed: {e}")
                
                finally:
                    # Move back to training device
                    model = model_cpu.to(device)
                    model.train()
            
            # Clean up
            del images, texts, loss
            if batch_ct % 10 == 0:
                torch.cuda.empty_cache()
        
        # End of epoch - make sure final gradients are applied
        if batch_ct % accumulation_steps != 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        # Save epoch checkpoint
        """
        epoch_path = os.path.join(model_save_dir, f"epoch{epoch+1:03d}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved epoch checkpoint to {epoch_path}")
        """
    
    # Final save
    final_path = os.path.join(model_save_dir, 'final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")


def train_batch(images, texts, model, device, criterion, optimizer, accumulation_steps=1):
    # Preprocess text if not already done
    if isinstance(texts, list) and isinstance(texts[0], str):
        texts = preprocess_text(texts, model)
        
        images, texts = images.to(device), texts.to(device)
        
        batch_size = images.shape[0]
    else:
        images = [img for img in images]
        batch_size = len(images)
        images = model.image_processor(images, return_tensors="pt").to(device)
        texts = model.tokenizer(texts, return_tensors="pt", padding=True, 
                               truncation=True, max_length=77).to(device)
    
    # Forward pass
    logits_per_image, logits_per_text = model(images, texts)
    
    # Create labels
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / accumulation_steps
    
    # Backward pass
    scaled_loss.backward()

    # Return unscaled loss for logging
    return loss.item()

def train_log(loss, example_ct, batch_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
args = parse_args()
model = model_pipeline(args)
    

