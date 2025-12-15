import torch
import cv2
import numpy as np
import os
from utils import compute_metrics, decode_targets

# =========================================================================
# 1. LOAD (For Fine-Tuning)
# =========================================================================
def load_weights_for_finetuning(model, checkpoint_path, device):
    """
    Loads weights from a pre-trained model but IGNORES the classifier head.
    Useful when changing vocab size (e.g., 62 chars -> 36 chars).
    """
    if not os.path.exists(checkpoint_path):
        print(f" Warning: Checkpoint not found at {checkpoint_path}. Starting from Scratch.")
        return model

    print(f" Loading Backbone weights from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different saving formats (state_dict vs full checkpoint)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 🛠️ THE TRICK: Filter out classifier weights
    # We keep ResNet (cnn) and Mamba (encoder) weights.
    # We drop 'classifier.weight' and 'classifier.bias' because shapes mismatch.
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
    
    # Check if we actually found matching keys
    if len(pretrained_dict) == 0:
        print(" Error: No matching weights found! Check layer names.")
        return model

    # Update the model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f" Loaded {len(pretrained_dict)} layers. Classifier initialized from scratch.")
    return model


# =========================================================================
# 2. INFER (Single Image Prediction)
# =========================================================================
def preprocess_for_inference(image, cfg):
    """
    Resizes, pads, and normalizes a single image for the model.
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    h, w = image.shape
    target_h, target_w = cfg.img_height, cfg.img_width
    
    # 2. Resize Height (32px)
    scale = target_h / h
    new_w = min(int(w * scale), target_w)
    image = cv2.resize(image, (new_w, target_h))
    
    # 3. Pad Width (to 320px)
    delta_w = target_w - new_w
    if delta_w > 0:
        image = cv2.copyMakeBorder(image, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=0)
        
    # 4. Normalize
    image = image.astype(np.float32) / 255.0
    image = np.stack([image, image, image], axis=0) # [3, H, W]
    
    mean = np.array(cfg.mean)[:, None, None]
    std = np.array(cfg.std)[:, None, None]
    image = (image - mean) / std
    
    # 5. Tensor
    return torch.from_numpy(image).float().unsqueeze(0) # [1, 3, 32, 320]

def predict_image(model, image, decoder, cfg, device):
    """
    End-to-End inference: Image -> Text
    """
    model.eval()
    
    # Preprocess
    input_tensor = preprocess_for_inference(image, cfg).to(device)
    
    # Forward
    with torch.no_grad():
        logits = model(input_tensor) # [1, T, V]
        
    # Decode
    text = decoder.decode_greedy(logits)[0]
    return text


# =========================================================================
# 3. EVAL (Full Validation Run)
# =========================================================================
def run_full_evaluation(model, data_loader, decoder, device, cfg):
    """
    Runs validation and prints "Best" and "Worst" predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print(" Starting Evaluation...")
    
    with torch.no_grad():
        for images, targets, target_lens in data_loader:
            images = images.to(device)
            
            # Forward
            logits = model(images)
            
            # Decode
            batch_preds = decoder.decode_greedy(logits)
            batch_targets = decode_targets(targets, target_lens, cfg.vocab)
            
            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)
            
    # Calculate Metrics
    cer, wer = compute_metrics(all_preds, all_targets)
    
    print(f"\n Evaluation Results:")
    print(f"   CER: {cer:.4f} ({(cer*100):.2f}%)")
    print(f"   WER: {wer:.4f} ({(wer*100):.2f}%)")
    
    # Show Failures
    print("\n Sample Failures:")
    shown = 0
    for p, t in zip(all_preds, all_targets):
        if p != t and shown < 5:
            print(f"   True: {t.ljust(15)} | Pred: {p}")
            shown += 1
            
    return cer