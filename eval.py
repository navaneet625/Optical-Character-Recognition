import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import jiwer
import os
import numpy as np

try:
    from fast_ctc_decode import beam_search
except ImportError:
    print("Warning: fast_ctc_decode not found. Beam search might fail.")
    def beam_search(probs, vocab, beam_size=10):
        return "" 

from models.ocr_model import MambaOCR
from configs.config import Config
from data.dataset import OCRDataset, load_data
from utils import CTCDecoder

def collate_fn(batch):
    images, targets, target_lens = zip(*batch)
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        if pad_w > 0:
            img = nn.functional.pad(img, (0, pad_w), value=0)
        padded_imgs.append(img)
    images = torch.stack(padded_imgs, dim=0)
    targets = torch.cat(targets)
    target_lens = torch.tensor(target_lens, dtype=torch.long)
    return images, targets, target_lens

# Helper function for decoding targets
def decode_targets(labels, lengths, vocab):
    decoded_texts = []
    current_idx = 0
    for length in lengths:
        label_indices = labels[current_idx : current_idx + length]
        text = "".join([vocab[i-1] for i in label_indices if i > 0 and (i-1) < len(vocab)])
        decoded_texts.append(text)
        current_idx += length
    return decoded_texts

# Helper function for computing metrics
def compute_metrics(predictions, targets):
    if not targets:
        return 0.0, 0.0 # Or handle as error/no data
    
    # jiwer expects lists of strings
    cer = jiwer.cer(targets, predictions)
    wer = jiwer.wer(targets, predictions)
    return cer, wer

def evaluate_model():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"--- Running Evaluation on {device} ---")
    
    # 1. Load Data
    all_paths, all_labels = load_data(cfg)
    # Using last 10% or fallback
    val_start = max(0, len(all_paths) - 100)
    if val_start > int(0.9 * len(all_paths)):
        val_start = int(0.9 * len(all_paths))

    val_dataset = OCRDataset(all_paths[val_start:], all_labels[val_start:], cfg, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Model
    model = MambaOCR(vocab_size=len(cfg.vocab)+1, 
                     cnn_out=cfg.cnn_out, 
                     n_layers=cfg.mamba_layers,
                     adapter_dim=cfg.adapter_dim).to(device)
    
    ckpt_path = cfg.best_model_path
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean DataParallel keys
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."): new_dict[k[7:]] = v
        else: new_dict[k] = v
        
    model.load_state_dict(new_dict, strict=True)
    model.eval()

    # 3. Predict
    decoder = CTCDecoder(cfg.vocab)
    all_preds = []
    all_targets = []
    
    print("Inferencing...")
    with torch.no_grad():
        for i, (images, targets, target_lengths) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            logits = model(images)
            
            # Beam Search for better WER
            # logits: [B, T, V]
            preds = decoder.decode_beam_search(logits, beam_width=5)
            decoded_targets = decode_targets(targets, target_lengths, cfg.vocab)
            
            all_preds.extend(preds)
            all_targets.extend(decoded_targets)
            
    cer, wer = compute_metrics(all_preds, all_targets)
    print(f"\nResults -> CER: {cer:.4f} | WER: {wer:.4f}")

if __name__ == "__main__":
    evaluate_model()