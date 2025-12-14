import torch
from torch.utils.data import DataLoader
from configs.config import Config
from models.ocr_model import MambaOCR
from data.dataset import OCRDataset
from utils import CTCDecoder
from utils_lifecycle import run_full_evaluation 

def collate_fn(batch):
    """
    Standard Collate: Pads images to max width in batch
    """
    images, targets, target_lens = zip(*batch)
    
    # Dynamic Padding
    max_w = max(img.shape[-1] for img in images)
    padded_imgs = []
    for img in images:
        w = img.shape[-1]
        pad_w = max_w - w
        if pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w), value=0)
        padded_imgs.append(img)
        
    images = torch.stack(padded_imgs, dim=0)
    # Flatten targets
    targets = torch.cat(targets)
    target_lens = torch.tensor(target_lens, dtype=torch.long)
    
    return images, targets, target_lens

def evaluate_model():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Running Evaluation on {device}")
    
    # 1. SETUP DATA
    # TRICK: We swap the 'train_csv' path with 'val_csv' 
    # so OCRDataset loads the Test Set instead of Training Set.
    print(f" Targeting Validation Set: {cfg.val_csv}")
    cfg.train_csv = cfg.val_csv 
    
    # Initialize Dataset (is_train=False disables augmentations)
    val_dataset = OCRDataset(cfg, is_train=False)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    print(f"Loaded {len(val_dataset)} validation images.")

    # 2. LOAD MODEL
    # Initialize Architecture
    model = MambaOCR(
        vocab_size=len(cfg.vocab)+1, 
        cnn_out=cfg.cnn_out, 
        n_layers=cfg.mamba_layers,
        adapter_dim=cfg.adapter_dim,
        lora_rank=cfg.lora_rank
    ).to(device)
    
    # Load Best Weights
    ckpt_path = cfg.best_model_path # e.g., 'best_mamba_ocr_uppercase.pth'
    
    if not torch.cuda.is_available():
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = torch.load(ckpt_path)
        
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    # Clean up DataParallel keys (remove "module.")
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print(f" Model weights loaded from: {ckpt_path}")
    except Exception as e:
        print(f" Error loading weights: {e}")
        return

    # 3. RUN EVALUATION
    # Uses the shared function from utils_lifecycle.py
    decoder = CTCDecoder(cfg.vocab)
    
    # This prints CER, WER, and sample failures automatically
    run_full_evaluation(model, val_loader, decoder, device, cfg)

if __name__ == "__main__":
    evaluate_model()