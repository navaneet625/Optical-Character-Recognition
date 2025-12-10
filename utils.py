import torch
import os
import jiwer
import matplotlib.pyplot as plt
import numpy as np

class CTCDecoder:
    """
    Decodes the raw output from the model (Logits) into text strings.
    Handles Greedy Decoding and removal of CTC blank tokens.
    """
    def __init__(self, vocab, blank_idx=0):
        self.vocab = vocab
        self.blank_idx = blank_idx

    def decode_greedy(self, log_probs):
        """
        Args:
            log_probs: Tensor of shape [Batch, Seq_Len, Vocab] or [Seq_Len, Batch, Vocab]
        Returns:
            List of decoded strings
        """
        # Ensure shape is [Batch, Seq_Len, Vocab]
        if log_probs.shape[0] != log_probs.shape[1] and log_probs.shape[2] == len(self.vocab) + 1:
             pass 
        
        # Get max probability indices
        arg_maxes = torch.argmax(log_probs, dim=2) # [B, T]
        
        decoded_batch = []
        
        for sequence in arg_maxes:
            decoded_text = []
            prev_char_idx = -1
            
            for char_idx in sequence:
                char_idx = char_idx.item()
                
                # CTC Logic: Only append if character is not blank AND not a repeat of the previous
                if char_idx != self.blank_idx and char_idx != prev_char_idx:
                    # Adjust index because vocab usually doesn't include blank at pos 0 in the list
                    # If vocab is "abc", blank=0, then a=1, b=2. vocab[char_idx-1] gets 'a'.
                    decoded_text.append(self.vocab[char_idx - 1])
                
                prev_char_idx = char_idx
            
            decoded_batch.append("".join(decoded_text))
            
        return decoded_batch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_metrics(predictions, targets):
    """
    Computes Character Error Rate (CER) and Word Error Rate (WER).
    Args:
        predictions: List of predicted strings.
        targets: List of ground truth strings.
    """
    cer = jiwer.cer(targets, predictions)
    wer = jiwer.wer(targets, predictions)
    return cer, wer

def save_checkpoint(model, optimizer, epoch, loss, save_dir="checkpoints", filename="best_model.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {start_epoch-1} with loss {loss:.4f}")
    return start_epoch, loss

def visualize_prediction(image_tensor, prediction, target=None):
    """
    Helper to visualize a single image and its prediction.
    """
    # Convert tensor back to numpy image
    # image_tensor: [1, H, W]
    img = image_tensor.squeeze().cpu().numpy()

    img = (img * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 2))
    plt.imshow(img, cmap='gray')
    title = f"Pred: {prediction}"
    if target:
        title += f" | True: {target}"
    plt.title(title)
    plt.axis('off')
    plt.show()