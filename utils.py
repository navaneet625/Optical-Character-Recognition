import os
import torch
import torch.nn as nn
import jiwer
import numpy as np

# -------------------------------------------------------------------
# Helper: Decode Targets for Validation
# -------------------------------------------------------------------
def decode_targets(targets, target_lengths, vocab):
    text_batch = []
    idx = 0
    for length in target_lengths:
        length = int(length)
        chars = []
        for t in targets[idx : idx + length]:
            if t > 0:
                chars.append(vocab[t - 1])
        text_batch.append("".join(chars))
        idx += length
    return text_batch

# -------------------------------------------------------------------
# Helper: Freeze Strategy (PEFT - Only Train Adapters/LoRA)
# -------------------------------------------------------------------
def apply_freeze_strategy(model, cfg):
    print("--- Applying Freeze Strategy ---")
    
    if isinstance(model, nn.DataParallel):
        cnn = model.module.cnn
        encoder = model.module.encoder
    else:
        cnn = model.cnn
        encoder = model.encoder

    # 1. Freeze CNN Backbone (ResNet)
    for p in cnn.parameters(): p.requires_grad = False
    
    # Unfreeze specific adapter layers
    for name, p in cnn.named_parameters():
        if "adapter" in name or "last_conv" in name or "layer4" in name or "backbone.7" in name:
            p.requires_grad = True

    # 2. Freeze Mamba Backbone
    for p in encoder.parameters(): p.requires_grad = False

    # Unfreeze LoRA & Projections
    for name, p in encoder.named_parameters():
        if "lora" in name or "project_in" in name:
            p.requires_grad = True

    # Summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f" Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class CTCDecoder:
    """Decodes raw model outputs into text."""
    def __init__(self, vocab: str, blank_idx: int = 0):
        self.vocab = vocab
        self.blank_idx = blank_idx 

    def decode_greedy(self, logits: torch.Tensor):
        """Greedy decode: Argmax -> Remove Blanks -> Remove Repeats"""
        x = logits.detach().cpu()
        if x.dim() != 3: raise ValueError("CTCDecoder expects [B, T, V] logits")

        idxs = torch.argmax(x, dim=2)
        decoded = []
        for seq in idxs:
            prev = None
            chars = []
            for ci in seq.tolist():
                if ci == self.blank_idx:
                    prev = ci
                    continue
                if ci == prev:
                    prev = ci
                    continue
                
                vocab_idx = ci - 1
                if 0 <= vocab_idx < len(self.vocab):
                    chars.append(self.vocab[vocab_idx])
                prev = ci
            decoded.append("".join(chars))
        return decoded

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

def compute_metrics(predictions, targets):
    """Computes CER and WER using jiwer."""
    preds = [str(p).strip() for p in predictions]
    refs = [str(t).strip() for t in targets]
    
    cer = jiwer.cer(refs, preds)
    wer = jiwer.wer(refs, preds)
    return cer, wer

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f" Checkpoint saved: {path}")