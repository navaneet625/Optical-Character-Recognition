import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from models.ocr_model import MambaOCR
from configs.config import Config
from data.dataset import OCRDataset, load_data
from utils import CTCDecoder, AverageMeter, compute_metrics, save_checkpoint

def decode_targets(targets, target_lengths, vocab):
    text_batch = []
    current_idx = 0
    for length in target_lengths:
        label_indices = targets[current_idx : current_idx + length]
        char_list = []
        for idx in label_indices:
            if idx > 0: # 0 is blank, skip it
                # Subtract 1 because vocab is 0-indexed in python string
                char_list.append(vocab[idx - 1])
        text_batch.append("".join(char_list))
        current_idx += length
    return text_batch


def collate_fn(batch):
    images, targets, target_lens = zip(*batch)

    # Pad Images Width to the max in the batch
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        if pad_w > 0:
            # Pad last dim (width) with 0
            img = nn.functional.pad(img, (0, pad_w), value=0)
        padded_imgs.append(img)

    images = torch.stack(padded_imgs, dim=0)
    targets = torch.cat(targets)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return images, targets, target_lens


def train():
    cfg = Config()

    # Check for CUDA
    use_cuda = torch.cuda.is_available() and cfg.device == 'cuda'
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on Device: {device}")

    # --- 1. DATA SETUP ---
    print(f"Loading data from: {cfg.data_dir}")
    all_img_paths, all_labels = load_data(cfg)

    if not all_img_paths:
        print("No images found! Check data_dir in configs/config.py")
        return

    full_dataset = OCRDataset(all_img_paths, all_labels, cfg, is_train=True)

    # Split Train/Val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    # Loaders
    workers = cfg.num_workers if use_cuda else 0
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=workers)

    # --- 2. MODEL SETUP ---
    # vocab_size + 1 because we need a slot for the CTC Blank token (Index 0)
    model = MambaOCR(vocab_size=len(cfg.vocab)+1,
                     cnn_out=cfg.cnn_out,
                     n_layers=cfg.mamba_layers,
                     adapter_dim=cfg.adapter_dim).to(device)

    # --- 3. FREEZING STRATEGY (CRITICAL FIX) ---
    print(">>> Setting up freezing...")

    # A. Freeze entire CNN Base first
    for param in model.cnn.parameters():
        param.requires_grad = False

    # B. Unfreeze adapters
    count_unfrozen = 0
    for name, param in model.cnn.named_parameters():
        if "lora" in name or "last_conv" in name:
            param.requires_grad = True
            count_unfrozen += 1

    # C. Unfreeze Mamba Projector
    for name, param in model.encoder.named_parameters():
        if "project_in" in name:
            param.requires_grad = True

    # Verify Trainable Params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    if count_unfrozen == 0:
        print(" WARNING: No CNN adapters were unfrozen! Check your layer naming.")

    # --- 4. OPTIMIZER & SCALER ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.learning_rate,
                           weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate,
                                              steps_per_epoch=len(train_loader),
                                              epochs=cfg.epochs)

    criterion = nn.CTCLoss(blank=cfg.blank_idx, zero_infinity=True)

    # Mixed Precision Scaler (initialize if on CUDA)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    decoder = CTCDecoder(vocab=cfg.vocab, blank_idx=cfg.blank_idx)

    # --- 5. TRAINING LOOP ---
    best_cer = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        model.cnn.eval()

        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # --- Forward Pass with Mixed Precision Check ---
            device_type = 'cuda' if use_cuda else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                preds = model(images) # [Batch, Time, Classes]

                # CTC Loss expects: [Time, Batch, Classes] (Log Softmax)
                log_probs = preds.log_softmax(2).permute(1, 0, 2)

                input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                print("Warning: Loss is NaN. Skipping batch.")
                continue

            # --- Backward Pass ---
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        print(f"Validating Epoch {epoch+1}...")
        val_cer, val_wer = validate(model, val_loader, decoder, device, cfg, use_cuda)
        print(f"Results - Loss: {loss_meter.avg:.4f} | Val CER: {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(model, optimizer, epoch, loss_meter.avg,
                            save_dir=cfg.checkpoint_dir, filename="best_mamba_ocr.pth")


def validate(model, loader, decoder, device, cfg, use_cuda):
    model.eval()
    all_preds = []
    all_targets = []

    device_type = 'cuda' if use_cuda else 'cpu'

    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)

            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                preds = model(images)

            # Decode Predictions
            decoded_preds = decoder.decode_greedy(preds)

            # Decode Targets (Ground Truth)
            decoded_targets = decode_targets(targets, target_lengths, cfg.vocab)

            all_preds.extend(decoded_preds)
            all_targets.extend(decoded_targets)

    if len(all_preds) > 0:
        print(f"Pred: {all_preds[0]} | True: {all_targets[0]}")

    return compute_metrics(all_preds, all_targets)

if __name__ == "__main__":
    train()