import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from configs.config import Config
from models.ocr_model import MambaOCR
from data.dataset import OCRDataset, load_data
from utils import CTCDecoder, AverageMeter, save_checkpoint, decode_targets, apply_freeze_strategy, compute_metrics


# -------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------
def collate_fn(batch):
    images, targets, target_lens = zip(*batch)

    # Dynamic Padding (Batch-wise)
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    
    for img in images:
        w = img.shape[-1]
        if w < max_w:
            img = nn.functional.pad(img, (0, max_w - w), value=0.0)
        padded_imgs.append(img)

    images = torch.stack(padded_imgs, dim=0)
    
    # Flatten targets for CTC Loss usage if needed, but here we keep 1D tensor of lengths
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets) if len(targets) > 0 else torch.tensor([], dtype=torch.long)

    return images, targets, target_lens


# -------------------------------------------------------------------
# Validation Loop
# -------------------------------------------------------------------
def validate(model, loader, decoder, device, cfg):
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            
            # Forward
            logits = model(images)
            
            # Greedy Decode
            decoded_preds = decoder.decode_greedy(logits)
            decoded_targets = decode_targets(targets, target_lengths, cfg.vocab)

            preds_list.extend(decoded_preds)
            targets_list.extend(decoded_targets)

    if preds_list:
        print(f" Sample: Pred='{preds_list[0]}' | True='{targets_list[0]}'")

    return compute_metrics(preds_list, targets_list)


# -------------------------------------------------------------------
# Train Loop
# -------------------------------------------------------------------
def train(cfg=None):
    if cfg is None:
        cfg = Config()
    
    print(f"--- Starting Training on {cfg.device} ---")
    
    # 1. Data
    paths, labels = load_data(cfg)
    full_ds = OCRDataset(paths, labels, cfg, is_train=True)
    
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True)

    # 2. Model
    model = MambaOCR(vocab_size=len(cfg.vocab)+1, 
                     cnn_out=cfg.cnn_out, 
                     n_layers=cfg.mamba_layers, 
                     adapter_dim=cfg.adapter_dim,
                     lora_rank=cfg.lora_rank).to(cfg.device)

    # 3. Freeze & Optimize
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    apply_freeze_strategy(model, cfg)
    
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                            lr=cfg.learning_rate, 
                            weight_decay=cfg.weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=cfg.epochs)
    
    criterion = nn.CTCLoss(blank=cfg.blank_idx, zero_infinity=True)
    decoder = CTCDecoder(cfg.vocab)
    
    # FP16 Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    # 4. Loop
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
            
            # Mixed Precision Context
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                # Forward: [B, T, V]
                outputs = model(images)
                log_probs = outputs.log_softmax(2).permute(1, 0, 2) # [T, B, V]
                
                input_lens = torch.full((images.size(0),), log_probs.size(0), 
                                        dtype=torch.long, device=cfg.device)
                
                loss = criterion(log_probs, targets, input_lens, target_lens)

            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            loss_meter.update(loss.item())
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
            
        # Validate
        val_cer, val_wer = validate(model, val_loader, decoder, cfg.device, cfg)
        print(f" Epoch {epoch+1} Results -> Loss: {loss_meter.avg:.4f} | CER: {val_cer:.4f}")
        
        # Save Best
        if val_cer < best_cer:
            best_cer = val_cer
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            save_checkpoint(model_to_save, optimizer, epoch, loss_meter.avg, 
                            save_dir=cfg.checkpoint_dir, filename="best_mamba_ocr.pth")


if __name__ == "__main__":
    train()
