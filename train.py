import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import glob
import os

# Project Imports
from models.ocr_model import MambaOCR
from configs.config import Config
from data.dataset import OCRDataset, load_data
from utils import CTCDecoder, AverageMeter, compute_metrics, save_checkpoint

# --- Helper: Convert indices back to text ---
def decode_targets(targets, target_lengths, vocab):
    text_batch = []
    current_idx = 0
    for length in target_lengths:
        label_indices = targets[current_idx : current_idx + length]
        char_list = []
        for idx in label_indices:
            if idx > 0: # 0 is blank
                char_list.append(vocab[idx - 1])
        text_batch.append("".join(char_list))
        current_idx += length
    return text_batch

def collate_fn(batch):
    """
    Returns tuple compatible with training loop unpacking
    """
    images, targets, target_lens = zip(*batch)

    # 1. Pad Images Width
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        if pad_w > 0:
            # Pad last dim (width)
            img = nn.functional.pad(img, (0, pad_w), value=0)
        padded_imgs.append(img)

    images = torch.stack(padded_imgs, dim=0) 
    targets = torch.cat(targets)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    # Return TUPLE so 'for img, targ, len in loader' works
    return images, targets, target_lens

def train():
    cfg = Config()
    device = torch.device(cfg.device)
    
    # --- 1. DATA SETUP ---
    print(f"Loading data from: {cfg.data_dir}")
    
    # Use the new load_data function
    all_img_paths, all_labels = load_data(cfg)
    
    if not all_img_paths:
        print("No images found! Please check your data_dir in configs/config.py")
        return

    full_dataset = OCRDataset(all_img_paths, all_labels, cfg, is_train=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cfg.num_workers)
    
    # --- 2. MODEL SETUP ---
    model = MambaOCR(vocab_size=len(cfg.vocab)+1, 
                     cnn_out=cfg.cnn_out, 
                     n_layers=cfg.mamba_layers).to(device)
    
    # --- 3. FREEZING STRATEGY (CRITICAL) ---
    print(">>> Setting up freezing...")
    
    # A. Freeze entire CNN Base
    for param in model.cnn.parameters():
        param.requires_grad = False
        
    # B. Unfreeze CNN Adapters
    for name, param in model.cnn.named_parameters():
        if "adapter" in name or "last_conv" in name:
            param.requires_grad = True

    # C. Freeze Mamba Base (Handled by PEFT, but double check)
    # LoRA usually handles this, but the 'project_in' layer needs manual unfreeze check
    for name, param in model.encoder.named_parameters():
        if "project_in" in name:
            param.requires_grad = True
            
    # Verify Trainable Params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    # --- 4. OPTIMIZER ---
    # ONLY pass trainable parameters to AdamW
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=cfg.learning_rate, 
                           weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, 
                                              steps_per_epoch=len(train_loader), epochs=cfg.epochs)
    
    criterion = nn.CTCLoss(blank=cfg.blank_idx, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    decoder = CTCDecoder(vocab=cfg.vocab, blank_idx=cfg.blank_idx)

    # --- 5. TRAINING LOOP ---
    best_cer = float('inf')
    
    for epoch in range(cfg.epochs):
        model.train()
        # Important: Keep BatchNorms in Eval mode if base CNN is frozen
        model.cnn.eval() 
        # But we need Adapters to be in train mode. 
        # Since they are nn.Conv2d, train/eval mostly affects Dropout/BN. 
        # Our adapter has no BN/Dropout, so .eval() on CNN is safe and preferred for frozen ResNet.
        
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            with torch.amp.autocast('cuda'):
                preds = model(images)
                log_probs = preds.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
        
        # Validation
        print(f"Validating Epoch {epoch+1}...")
        val_cer, val_wer = validate(model, val_loader, decoder, device, cfg)
        print(f"Results - Loss: {loss_meter.avg:.4f} | Val CER: {val_cer:.4f}")
        
        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(model, optimizer, epoch, loss_meter.avg, save_dir=cfg.checkpoint_dir, filename="best_mamba_ocr.pth")
            
def validate(model, loader, decoder, device, cfg):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                preds = model(images) 
            decoded_preds = decoder.decode_greedy(preds)
            decoded_targets = decode_targets(targets, target_lengths, cfg.vocab)
            all_preds.extend(decoded_preds)
            all_targets.extend(decoded_targets)
    if len(all_preds) > 0:
        print(f"Pred: {all_preds[0]} | True: {all_targets[0]}")
    return compute_metrics(all_preds, all_targets)

if __name__ == "__main__":
    train()