import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from models.ocr_model import MambaOCR
from configs.config import Config
from data.dataset import OCRDataset
from utils import CTCDecoder, decode_targets

def visualize_failures():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Visualizing Failures on {device} ")

    # 1. SETUP DATA (Targeting Validation Set)
    print(f"Targeting Validation Set: {cfg.val_csv}")
    # Trick: Swap train_csv with val_csv to load the test set
    cfg.train_csv = cfg.val_csv 
    
    # Initialize Dataset
    val_dataset = OCRDataset(cfg, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Loaded {len(val_dataset)} images.")

    # 2. LOAD MODEL
    print("Loading Model Weights...")
    model = MambaOCR(
        vocab_size=len(cfg.vocab)+1, 
        cnn_out=cfg.cnn_out, 
        n_layers=cfg.mamba_layers, 
        adapter_dim=cfg.adapter_dim,
        lora_rank=cfg.lora_rank 
    ).to(device)
    
    # Load Weights (Handling 'best_mamba_ocr_uppercase.pth')
    ckpt_path = cfg.best_model_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Clean DataParallel keys
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()
    
    decoder = CTCDecoder(vocab=cfg.vocab, blank_idx=cfg.blank_idx)
    failures = []
    
    print("🕵️ Hunting for errors...")
    
    # 3. INFERENCE LOOP
    with torch.no_grad():
        for i, (img, label_indices, label_len) in enumerate(val_loader):
            img = img.to(device)
            
            # Forward
            preds = model(img)
            
            # Decode Prediction
            pred_text = decoder.decode_greedy(preds)[0]
            
            # Decode Ground Truth (Using shared utility)
            # label_indices is [1, SeqLen]
            true_text = decode_targets(label_indices.flatten(), label_len, cfg.vocab)[0]
            
            # Compare
            if pred_text != true_text:
                # Prepare Image for Plotting
                # [1, 3, H, W] -> [H, W, 3]
                img_t = img.squeeze(0).permute(1, 2, 0).cpu()
                
                # Unnormalize using Config stats
                mean = torch.tensor(cfg.mean).view(1, 1, 3)
                std = torch.tensor(cfg.std).view(1, 1, 3)
                img_t = img_t * std + mean
                img_np = img_t.numpy().clip(0, 1)
                
                failures.append((img_np, pred_text, true_text))
                print(f"   Found Error: True='{true_text}' vs Pred='{pred_text}'")
                
            if len(failures) >= 9:
                break
    
    # 4. PLOTTING
    if len(failures) == 0:
        print(" No errors found in the first batch.")
        return

    print(f"Found {len(failures)} errors. Saving plot...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    # Handle case where fewer than 9 failures exist
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(failures):
            img, pred, true = failures[idx]
            ax.imshow(img)
            ax.set_title(f"True: {true}\nPred: {pred}", color='red', fontsize=10, weight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("error_analysis.png")
    plt.show()
    print("Saved failure grid to 'error_analysis.png'")

if __name__ == "__main__":
    visualize_failures()