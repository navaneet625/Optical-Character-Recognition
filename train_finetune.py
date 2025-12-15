import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from configs.config import Config
from models.ocr_model import MambaOCR
from data.dataset import OCRDataset 
from utils import CTCDecoder, AverageMeter, save_checkpoint, decode_targets, compute_metrics

# --- 1. ROBUST COLLATE FN ---
def collate_fn(batch):
    batch_data = list(zip(*batch))
    images = batch_data[0]
    targets = batch_data[1]
    
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    for img in images:
        w = img.shape[-1]
        if w < max_w:
            img = nn.functional.pad(img, (0, max_w - w), value=0.0)
        padded_imgs.append(img)
    images = torch.stack(padded_imgs, dim=0)
    
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets) if len(targets) > 0 else torch.tensor([], dtype=torch.long)

    return images, targets, target_lens


def apply_freeze_strategy(model):
    
    if isinstance(model, nn.DataParallel):
        raw_model = model.module
    else:
        raw_model = model

    # 1. Freeze Bottom Layers (Edges/Textures) - Keep these stable
    for p in raw_model.parameters():
        p.requires_grad = False

    # 2. Unfreeze ResNet Layer 4 (High-Level Shapes like '3' vs '9')
    # This is the FIX for your specific errors
    for name, p in raw_model.cnn.named_parameters():
        if "layer4" in name or "adapter" in name or "last_conv" in name:
            p.requires_grad = True

    # 3. Unfreeze Mamba & Head (Logic)
    for name, p in raw_model.encoder.named_parameters():
        if "lora" in name or "project_in" in name or "norm" in name:
            p.requires_grad = True

    for name, child in raw_model.named_children():
        if "head" in name or "fc" in name:
            for p in child.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f" Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def validate(model, loader, decoder, device, cfg):
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            logits = model(images)
            decoded_preds = decoder.decode_greedy(logits)
            decoded_targets = decode_targets(targets, target_lengths, cfg.vocab)
            preds_list.extend(decoded_preds)
            targets_list.extend(decoded_targets)
    return compute_metrics(preds_list, targets_list)

def train_finetune():
    cfg = Config()
    print(f"---  Starting FINE-TUNING on {cfg.device} ---")
    
    # Dataset
    full_ds = OCRDataset(cfg, is_train=True)
    train_size = int(0.9 * len(full_ds)) # 90% train, 10% val
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    print(f" Data: {len(train_ds)} Train | {len(val_ds)} Val")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=cfg.num_workers)

    # Model Init
    model = MambaOCR(vocab_size=len(cfg.vocab)+1, 
                     cnn_out=cfg.cnn_out, n_layers=cfg.mamba_layers, 
                     adapter_dim=cfg.adapter_dim, lora_rank=cfg.lora_rank).to(cfg.device)

    # Load Pre-trained Weights
    if os.path.exists(cfg.pretrained_weights):
        print(f" Loading Weights: {cfg.pretrained_weights}")
        checkpoint = torch.load(cfg.pretrained_weights, map_location=cfg.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(clean_state_dict, strict=True)
        except:
            print(" Strict load failed. Retrying with strict=False...")
            model.load_state_dict(clean_state_dict, strict=False)
    else:
        print(" PRE-TRAINED WEIGHTS NOT FOUND! Check Config.")
        return

    # Apply Freeze
    apply_freeze_strategy(model)

    # Optimizer (Only trainable params)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=cfg.learning_rate, 
                            weight_decay=cfg.weight_decay)
    
    # Scheduler: Cosine Annealing (Smooth decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    criterion = nn.CTCLoss(blank=cfg.blank_idx, zero_infinity=True)
    decoder = CTCDecoder(cfg.vocab)
    scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision)

    best_cer = float("inf")
    
    for epoch in range(cfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for images, targets, target_lens in pbar:
            images = images.to(cfg.device)
            targets = targets.to(cfg.device)
            target_lens = target_lens.to(cfg.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
                outputs = model(images)
                log_probs = outputs.log_softmax(2).permute(1, 0, 2)
                input_lens = torch.full((images.size(0),), log_probs.size(0), 
                                        dtype=torch.long, device=cfg.device)
                loss = criterion(log_probs, targets, input_lens, target_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            
            loss_meter.update(loss.item())
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        
        scheduler.step()
        
        # Validate
        val_cer, val_wer = validate(model, val_loader, decoder, cfg.device, cfg)
        print(f" Epoch {epoch+1} -> Loss: {loss_meter.avg:.4f} | Val CER: {val_cer:.4f}")
        
        # Save Best
        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(model, optimizer, epoch, loss_meter.avg, 
                            save_dir=cfg.checkpoint_dir, filename="finetuned_best.pth")

if __name__ == "__main__":
    train_finetune()