import torch
import matplotlib.pyplot as plt
from models.ocr_model import MambaOCR
from configs.config import Config
from data.dataset import OCRDataset, load_data
from torch.utils.data import DataLoader
from utils import CTCDecoder

def visualize_failures():
    cfg = Config()
    device = torch.device(cfg.device)

    print("Loading Validation Data...")
    all_img_paths, all_labels = load_data(cfg)
    val_start = int(0.9 * len(all_img_paths))
    val_dataset = OCRDataset(all_img_paths[val_start:], all_labels[val_start:], cfg, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Loading Checkpoint...")
    model = MambaOCR(vocab_size=len(cfg.vocab)+1, 
                     cnn_out=cfg.cnn_out, 
                     n_layers=cfg.mamba_layers, 
                     adapter_dim=cfg.adapter_dim).to(device)
    checkpoint = torch.load("checkpoints/best_mamba_ocr.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
    model.eval()
    
    decoder = CTCDecoder(vocab=cfg.vocab, blank_idx=cfg.blank_idx)
    
    failures = []
    
    print("Hunting for errors...")
    with torch.no_grad():
        for i, (img, label_idx, _) in enumerate(val_loader):
            img = img.to(device)
            preds = model(img)
            pred_text = decoder.decode_greedy(preds)[0]
            
            true_indices = label_idx[0]
            true_text = "".join([cfg.vocab[idx-1] for idx in true_indices if idx > 0])
            
            if pred_text != true_text:
                # [1, 3, H, W] -> [3, H, W] -> [H, W, 3]
                img_t = img.squeeze(0).permute(1, 2, 0).cpu()
                
                # Unnormalize (approx)
                mean = torch.tensor(cfg.mean).view(1, 1, 3)
                std = torch.tensor(cfg.std).view(1, 1, 3)
                img_t = img_t * std + mean
                img_np = img_t.numpy().clip(0, 1)
                
                failures.append((img_np, pred_text, true_text))
                
            if len(failures) >= 9:
                break
    
    # Plot
    print(f"Found {len(failures)} errors. Plotting...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    for ax, (img, pred, true) in zip(axes.flatten(), failures):
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true}\nPred: {pred}", color='red', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("error_analysis.png")
    plt.show()

if __name__ == "__main__":
    visualize_failures()