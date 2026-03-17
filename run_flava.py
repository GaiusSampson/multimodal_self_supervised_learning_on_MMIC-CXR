import os
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from train import load_data, load_clip
from zero_shot_flava import (
    run_softmax_eval_flava,
    load_test_data_flava,
    make_true_labels
)
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str,
                        default='data/cxr.h5')
    parser.add_argument('--txt_filepath', type=str,
                        default='data/mimic_impressions.csv')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--effective_batch_size', type=int, default=96)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--minimum_lr', type=float, default=1e-7)
    parser.add_argument('--head_lr', type=float, default=1e-5)
    parser.add_argument('--minimum_head_lr', type=float, default=1e-7)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--seed', type=int, default=1243)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--model_name', type=str,
                        default="pt-imp")
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--use_biobert', action='store_true',
                        help='Use BioClinicalBERT text encoder')
    parser.add_argument('--biobert_model', type=str,
                        default='emilyalsentzer/Bio_ClinicalBERT',
                        choices=[
                            'emilyalsentzer/Bio_ClinicalBERT',
                            'emilyalsentzer/Bio_Discharge_Summary_BERT',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                        ])
    # Loss weights set to 0 to disable individual losses
    parser.add_argument('--lambda_contrastive', type=float, default=1.5,
                        help='Weight for contrastive loss')
    parser.add_argument('--lambda_itm', type=float, default=0.2,
                        help='Weight for image-text matching loss')
    parser.add_argument('--lambda_mlm', type=float, default=0.1,
                        help='Weight for masked language modelling loss')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='Probability of masking each token for MLM')
    args = parser.parse_args()
    return args



class FLAVALossHeads(nn.Module):
    """
    Additional loss heads for full FLAVA training:
    - ITM head: binary classification for image-text matching
    - MLM head: token prediction for masked language modelling
    Both operate on outputs from the multimodal encoder.
    """
    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522):
        super().__init__()

        # ITM head binary: does this image match this text?
        self.itm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2)  # match / no match
        )

        # MLM head to predict masked tokens
        # vocab_size 30522 = standard BERT vocabulary
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

        # Initialise MLM output layer with BERT weights
        nn.init.normal_(self.mlm_head[-1].weight, std=0.02)
        nn.init.zeros_(self.mlm_head[-1].bias)

        # Same for ITM for consistency
        nn.init.normal_(self.itm_head[-1].weight, std=0.02)
        nn.init.zeros_(self.itm_head[-1].bias)

    def forward_itm(self, multimodal_cls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multimodal_cls: [B, hidden_size] CLS token from multimodal encoder
        Returns:
            logits: [B, 2]
        """
        return self.itm_head(multimodal_cls)

    def forward_mlm(self,
                    multimodal_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multimodal_seq: [B, seq_len, hidden_size] full sequence
                            from multimodal encoder
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        return self.mlm_head(multimodal_seq)



def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer,
    mlm_probability: float = 0.15
) -> tuple:
    """
    Prepare masked tokens for MLM following BERT masking strategy:
    - 80% of selected tokens replaced with [MASK]
    - 10% replaced with random token
    - 10% left unchanged
    Only masks non-special tokens (not [CLS], [SEP], [PAD]).

    Args:
        input_ids: [B, seq_len] token ids
        tokenizer: HuggingFace tokenizer
        mlm_probability: Fraction of tokens to mask

    Returns:
        masked_input_ids: [B, seq_len] with masks applied
        mlm_labels: [B, seq_len] original ids at masked positions,
                    -100 elsewhere (ignored by CrossEntropyLoss)
    """
    labels = input_ids.clone()
    device = input_ids.device

    # Build probability matrix
    probability_matrix = torch.full(labels.shape, mlm_probability,
                                    device=device)

    # Don't mask special tokens
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id
    ]:
        if special_id is not None:
            special_tokens_mask |= (input_ids == special_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Sample which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only compute loss on masked tokens, ignore rest
    labels[~masked_indices] = -100

    # 80% of time replace with [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool()
        & masked_indices
    )
    input_ids_masked = input_ids.clone()
    input_ids_masked[indices_replaced] = tokenizer.mask_token_id

    # 10% of time replace with random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape,
        dtype=torch.long, device=device
    )
    input_ids_masked[indices_random] = random_words[indices_random]

    # Remaining 10% kept unchanged
    return input_ids_masked, labels



def create_itm_batch(
    pixel_values: torch.Tensor,
    text_tokens: dict,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Create a batch for ITM training using hard negatives.
    Hard negatives are the most similar mismatched pairs
    in the batch — harder for the model to distinguish
    than random mismatches.

    For a batch of size B, creates 2B samples:
    - B positive pairs  (label=1, matching)
    - B negative pairs  (label=0, mismatched)

    Negative pairs are selected by finding the most similar
    image for each text (and vice versa) that isn't the
    correct match, making them harder negatives.

    Args:
        pixel_values: [B, 3, 224, 224]
        text_tokens: dict with input_ids, attention_mask [B, seq_len]
        image_features: normalised image embeddings [B, embed_dim]
        text_features: normalised text embeddings [B, embed_dim]
        device: torch device

    Returns:
        itm_pixel_values, itm_text_tokens, itm_labels
    """
    B = pixel_values.shape[0]

    # Similarity matrix to find hard negatives
    with torch.no_grad():
        sim = image_features @ text_features.T  # [B, B]
        # Zero out diagonal (correct pairs)
        sim.fill_diagonal_(float('-inf'))

    # For each image, find most similar text (hard negative)
    hard_neg_text_idx = sim.argmax(dim=1)  # [B]

    
    #we build only the negative half here
    # and run the two halves (pos / neg) as separate forward passes,
    # keeping peak activations at B rather than 2B.

    # Positive: original pairs (pixel_values, text_tokens) — returned as-is
    # Negative: correct image paired with the hard-negative text
    neg_text_tokens = {
        k: v[hard_neg_text_idx] for k, v in text_tokens.items()
    }

    itm_labels_pos = torch.ones(B, dtype=torch.long, device=device)
    itm_labels_neg = torch.zeros(B, dtype=torch.long, device=device)

    return (
        pixel_values, text_tokens, itm_labels_pos,   # positive half
        pixel_values, neg_text_tokens, itm_labels_neg  # negative half
    )



def get_multimodal_features(model, pixel_values, text_tokens, device):
    image_outputs = model.flava.get_image_features(pixel_values=pixel_values)
    image_sequence = image_outputs.last_hidden_state  # [B, 197, 768]

    text_outputs = model.flava.get_text_features(**text_tokens)
    text_sequence = text_outputs.last_hidden_state  # [B, seq_len, 768]

    B = pixel_values.shape[0]

    image_attention = torch.ones(
        B, image_sequence.shape[1], dtype=torch.long, device=device
    )
    combined_attention = torch.cat(
        [image_attention, text_tokens['attention_mask']], dim=1
    )  # [B, 197+seq_len]

    # Prepend 1 for the CLS token
    cls_attention = torch.ones(B, 1, dtype=torch.long, device=device)
    full_attention = torch.cat([cls_attention, combined_attention], dim=1)
    # Shape: [B, 275]

    # Pass 2D
    multimodal_output = model.flava.multimodal_model(
        hidden_states=torch.cat([image_sequence, text_sequence], dim=1),
        attention_mask=full_attention  # 2D [B, 275], not extended 4D
    )

    return multimodal_output.last_hidden_state  # [B, 1+197+seq_len, 768]



def model_pipeline(config, verbose=0):
    if config.run_dir is None:
        config.run_dir = f"./runs/{config.model_name}"
    writer = SummaryWriter(config.run_dir)

    model, loss_heads, data_loader, device, optimizers, scheduler, scheduler_heads = make(config)

    train(
        model, loss_heads, data_loader, device,
        optimizers, config, writer, scheduler, scheduler_heads
    )

    # Final save
    model_path = os.path.join(
        config.save_dir, config.model_name, 'checkpoint.pt'
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to {model_path}")

    if verbose:
        print(model)
    return model


def make(config):
    # Load data with FLAVA preprocessing (no normalization)
    data_loader, device = load_data(
        config.cxr_filepath,
        config.txt_filepath,
        batch_size=config.batch_size,
        pretrained=False,
        column="impression",
        flava=True
    )

    # Build FLAVA model
    model = load_clip(
        model_path=None,
        pretrained=False,
        context_length=config.context_length,
        flava=True,
        use_biobert=config.use_biobert,
        biobert_model=config.biobert_model
    )
    model = model.float().to(device)
    model.train()
    print(f"Loaded FLAVA with: "
          f"{'BioClinicalBERT' if config.use_biobert else 'Standard BERT'}")

    # Get vocab size from tokenizer for MLM head
    vocab_size = len(model.tokenizer)

    # Loss heads (ITM + MLM)
    loss_heads = FLAVALossHeads(
        hidden_size=768,
        vocab_size=vocab_size
    ).to(device)



    # Separate optimizers so we can control lr per component
    # FLAVA pretrained weights get lower lr
    flava_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
    ]
    head_params = list(loss_heads.parameters())

    criterion_contrastive = nn.CrossEntropyLoss().to(device)
    criterion_itm = nn.CrossEntropyLoss().to(device)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    if config.optimizer == "adam":
        optimizer_model = optim.AdamW(
            flava_params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        optimizer_heads = optim.AdamW(
            head_params,
            lr=config.head_lr,  
            weight_decay=config.weight_decay
        )
    else:
        optimizer_model = optim.SGD(
            flava_params, lr=config.lr, momentum=config.momentum
        )
        optimizer_heads = optim.SGD(
            head_params, lr=config.head_lr, momentum=config.momentum
        )

    optimizers = {
        'model': optimizer_model,
        'heads': optimizer_heads,
        'contrastive': criterion_contrastive,
        'itm': criterion_itm,
        'mlm': criterion_mlm,
    }

    batches_per_epoch = 377112 / config.batch_size
    total_batches = (
        batches_per_epoch * config.epochs
        // (config.effective_batch_size / config.batch_size)
    )

    scheduler = CosineAnnealingLR(
        optimizer_model,
        T_max=total_batches,
        eta_min=config.minimum_lr
    )

    scheduler_heads = CosineAnnealingLR(
    optimizer_heads,
    T_max=total_batches,
    eta_min=config.minimum_head_lr
    )

    return model, loss_heads, data_loader, device, optimizers, scheduler, scheduler_heads

def get_best_threshold(y_true_col, y_pred_col):
    """Find threshold that maximises MCC for a single label."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_mcc = -1.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_pred_col >= t).astype(int)
        mcc = matthews_corrcoef(y_true_col, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
    return best_thresh

def train(model, loss_heads, loader, device, optimizers,
          config, logger, scheduler=None, scheduler_heads=None):

    model_save_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    accumulation_steps = max(
        1, config.effective_batch_size // config.batch_size
    )
    print(f"Physical batch size:   {config.batch_size}")
    print(f"Accumulation steps:    {accumulation_steps}")
    print(f"Effective batch size:  {config.batch_size * accumulation_steps}")
    print(f"Loss weights — "
          f"Contrastive: {config.lambda_contrastive}  "
          f"ITM: {config.lambda_itm}  "
          f"MLM: {config.lambda_mlm}")

    cxr_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    cxr_pair_template = ("{}", "no {}")

    # Load validation data
    test_loader = load_test_data_flava("test_data/chexpert_val.h5")
    y_true = make_true_labels("test_data/val.csv", cxr_labels)
    print("Validation data loaded")

    best_models = {
        i: {
            "Filepath": os.path.join(model_save_dir, f"Best_{i}.pt"),
            "Mean_AUC": 0
        } for i in range(10)
    }

    example_ct = 0
    batch_ct = 0
    running = {
        'total': 0.0,
        'contrastive': 0.0,
        'itm': 0.0,
        'mlm': 0.0
    }

    for epoch in range(config.epochs):
        print(f"\nStarting epoch {epoch + 1}/{config.epochs}")

         # Freeze backbone encoders after epoch 1
        if epoch >= 3:
            for param in model.flava.image_model.parameters():
                param.requires_grad = False
            for param in model.flava.text_model.parameters():
                param.requires_grad = False
            print("Backbone encoders frozen only training multimodal layers and heads")


        model.train()
        loss_heads.train()

        optimizers['model'].zero_grad()
        optimizers['heads'].zero_grad()

        for data in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            images = data['img']
            texts = data['txt']

            losses = train_batch(
                config=config,
                images=images,
                texts=texts,
                model=model,
                loss_heads=loss_heads,
                device=device,
                optimizers=optimizers,
                accumulation_steps=accumulation_steps
            )

            example_ct += len(images)
            batch_ct += 1

            for k in running:
                running[k] += losses.get(k, 0.0)

            # Optimizer step every accumulation_steps
            if batch_ct % accumulation_steps == 0:
                # Gradient clipping to stabilise FLAVA training
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                torch.nn.utils.clip_grad_norm_(
                    loss_heads.parameters(), max_norm=1.0
                )
                optimizers['model'].step()
                optimizers['heads'].step()
                if scheduler is not None:
                    scheduler.step()
                if scheduler_heads is not None:
                    scheduler_heads.step()
                optimizers['model'].zero_grad()
                optimizers['heads'].zero_grad()

            # Logging
            if batch_ct % config.log_interval == 0:
                for k, v in running.items():
                    avg = v / config.log_interval
                    logger.add_scalar(f"Loss/{k}", avg, batch_ct)
                print(
                    f"loss after {example_ct:06d} examples "
                    f"{running['total']/config.log_interval:.3f}\n"
                    f"contrastive loss {running['contrastive']/config.log_interval:.3f}  "
                    f"itm loss {running['itm']/config.log_interval:.3f}  "
                    f"mlm loss {running['mlm']/config.log_interval:.3f}"
                )
                running = {k: 0.0 for k in running}

            # Validation checkpoint
            if batch_ct % config.save_interval == 0:
                # Flush remaining gradients
                if batch_ct % accumulation_steps != 0:
                    optimizers['model'].step()
                    optimizers['heads'].step()
                    if scheduler is not None:
                        scheduler.step()
                    if scheduler_heads is not None:
                        scheduler_heads.step()
                    optimizers['model'].zero_grad()
                    optimizers['heads'].zero_grad()

                model_cpu = model.cpu().eval()
                loss_heads_cpu = loss_heads.cpu().eval()

                try:
                    with torch.no_grad():
                        y_pred = run_softmax_eval_flava(
                            model_cpu, test_loader,
                            cxr_labels, cxr_pair_template
                        )
                    stats = evaluate(y_pred, y_true, cxr_labels)
                    mean_auc = stats.mean(1)[0]

                    # Find optimal threshold per label then compute MCC
                    best_thresholds = [
                        get_best_threshold(y_true[:, i], y_pred[:, i])
                        for i in range(len(cxr_labels))
                    ]
                    mean_mcc = np.mean([
                        matthews_corrcoef(
                            y_true[:, i],
                            (y_pred[:, i] >= best_thresholds[i]).astype(int)
                        )
                        for i in range(len(cxr_labels))
                    ])

                    print(f"Validation AUC: {mean_auc:.4f}  MCC: {mean_mcc:.4f}")
                    logger.add_scalar("AUC/val", mean_auc, batch_ct)
                    logger.add_scalar("MCC/val", mean_mcc, batch_ct)

                    # Top-10 tracking by AUC
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
                    import traceback
                    traceback.print_exc()
                finally:
                    model = model_cpu.to(device).train()
                    loss_heads = loss_heads_cpu.to(device).train()

            del images, texts
            if batch_ct % 10 == 0:
                torch.cuda.empty_cache()

        if batch_ct % accumulation_steps != 0:
            optimizers['model'].step()
            optimizers['heads'].step()
            if scheduler is not None:
                scheduler.step()
            if scheduler_heads is not None:
                scheduler_heads.step()
            optimizers['model'].zero_grad()
            optimizers['heads'].zero_grad()


    final_path = os.path.join(model_save_dir, 'final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'heads_state_dict': loss_heads.state_dict(),
    }, final_path)
    print(f"Training complete! Saved to {final_path}")



def train_batch(config, images, texts, model, loss_heads,
                device, optimizers, accumulation_steps=1,
                warmup_scale=1.0):

    tokens = model.tokenizer(
        texts, padding=True, truncation=True,
        max_length=model.context_length, return_tensors="pt"
    )
    text_tokens = {k: v.to(device) for k, v in tokens.items()}

    pil_images = [to_pil_image(img.cpu()) for img in images]
    processed = model.processor(images=pil_images, return_tensors="pt")
    pixel_values = processed['pixel_values'].to(device)

    raw_tensors = torch.stack([
        torch.from_numpy(np.array(p)).permute(2, 0, 1).float() / 255.0
        for p in pil_images
    ])
    del pil_images, processed

    batch_size = pixel_values.shape[0]

    # Detached features for hard-negative mining only
    with torch.no_grad():
        image_features_det = model.encode_image(pixel_values)
        text_features_det  = model.encode_text(text_tokens)
        image_features_norm_det = image_features_det / image_features_det.norm(dim=-1, keepdim=True)
        text_features_norm_det  = text_features_det  / text_features_det.norm(dim=-1, keepdim=True)

    # Contrastive loss backward immediately, free graph
    image_features = model.encode_image(pixel_values)
    text_features  = model.encode_text(text_tokens)

    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm  = text_features  / text_features.norm(dim=-1, keepdim=True)

    logit_scale      = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features_norm @ text_features_norm.mT
    logits_per_text  = logit_scale * text_features_norm  @ image_features_norm.mT

    labels = torch.arange(batch_size, device=device)
    loss_contrastive = (
        optimizers['contrastive'](logits_per_image, labels) +
        optimizers['contrastive'](logits_per_text,  labels)
    ) / 2

    # Backward NOW frees the contrastive graph before ITM forward passes
    (config.lambda_contrastive * loss_contrastive / accumulation_steps).backward()

    loss_contrastive_val = loss_contrastive.item()
    del logits_per_image, logits_per_text, image_features, text_features
    del image_features_norm, text_features_norm, loss_contrastive
    torch.cuda.empty_cache()

    # ITM loss two B-size passes, each freed immediately
    raw_tensors_gpu = raw_tensors.to(device)
    del raw_tensors

    (pos_pv, pos_tt, pos_labels,
     neg_pv, neg_tt, neg_labels) = create_itm_batch(
        pixel_values=raw_tensors_gpu,
        text_tokens=text_tokens,
        image_features=image_features_norm_det,
        text_features=text_features_norm_det,
        device=device
    )
    del image_features_norm_det, text_features_norm_det

    # Positive ITM backward immediately
    pos_multimodal  = model.encode_multimodal(pos_pv, pos_tt)
    pos_itm_logits  = loss_heads.forward_itm(pos_multimodal[:, 0, :])
    loss_itm_pos    = optimizers['itm'](pos_itm_logits, pos_labels)
    (config.lambda_itm * warmup_scale * loss_itm_pos / 2 / accumulation_steps).backward()
    loss_itm_pos_val = loss_itm_pos.item()
    del pos_multimodal, pos_itm_logits, loss_itm_pos
    torch.cuda.empty_cache()

    # Negative ITM backward immediately
    neg_multimodal  = model.encode_multimodal(neg_pv, neg_tt)
    neg_itm_logits  = loss_heads.forward_itm(neg_multimodal[:, 0, :])
    loss_itm_neg    = optimizers['itm'](neg_itm_logits, neg_labels)
    (config.lambda_itm * warmup_scale * loss_itm_neg / 2 / accumulation_steps).backward()
    loss_itm_neg_val = loss_itm_neg.item()
    del neg_multimodal, neg_itm_logits, loss_itm_neg
    torch.cuda.empty_cache()

    loss_itm_val = (loss_itm_pos_val + loss_itm_neg_val) / 2

    # MLM loss backward immediately, free graph
    masked_input_ids, mlm_labels = mask_tokens(
        input_ids=text_tokens['input_ids'],
        tokenizer=model.tokenizer,
        mlm_probability=config.mlm_probability
    )
    masked_text_tokens = {'input_ids': masked_input_ids,
                          'attention_mask': text_tokens['attention_mask']}
    if 'token_type_ids' in text_tokens:
        masked_text_tokens['token_type_ids'] = text_tokens['token_type_ids']

    mlm_multimodal  = model.encode_multimodal(raw_tensors_gpu, masked_text_tokens)
    del raw_tensors_gpu
    text_multimodal = mlm_multimodal[:, 1 + 197:, :].contiguous()
    del mlm_multimodal
    mlm_logits = loss_heads.forward_mlm(text_multimodal)
    del text_multimodal
    loss_mlm = optimizers['mlm'](
        mlm_logits.reshape(-1, mlm_logits.shape[-1]),
        mlm_labels.reshape(-1)
    )
    del mlm_logits
    (config.lambda_mlm * warmup_scale * loss_mlm / accumulation_steps).backward()
    loss_mlm_val = loss_mlm.item()
    del loss_mlm
    torch.cuda.empty_cache()

    # Scalar for logging only (no graph attached)
    loss_total_val = (
        config.lambda_contrastive * loss_contrastive_val +
        config.lambda_itm         * loss_itm_val +
        config.lambda_mlm         * loss_mlm_val
    )

    return {
        'total':       loss_total_val,
        'contrastive': loss_contrastive_val,
        'itm':         loss_itm_val,
        'mlm':         loss_mlm_val,
    }


def save(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = model_pipeline(args)