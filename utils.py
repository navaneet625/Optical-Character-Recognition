import os
import torch
import torch.nn as nn
import jiwer
import matplotlib.pyplot as plt
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
# Helper: Freeze Strategy (PEFT)
# -------------------------------------------------------------------
def apply_freeze_strategy(model, cfg):
    """
    Freezes everything except Adapters, LoRA, and Projectors.
    """
    print("--- Applying Freeze Strategy ---")
    
    # Unwrap DataParallel if needed
    if isinstance(model, nn.DataParallel):
        cnn = model.module.cnn
        encoder = model.module.encoder
    else:
        cnn = model.cnn
        encoder = model.encoder

    # 1. Freeze CNN Backbone
    for p in cnn.parameters(): p.requires_grad = False
    
    # Unfreeze Adapters
    # Unfreeze Adapters & Layer 4
    # Note: In ResNetFeatureExtractor, backbone is nn.Sequential.
    # layer4 corresponds to index 7 in the sequence:
    # 0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4
    for name, p in cnn.named_parameters():
        if "adapter" in name or "last_conv" in name or "layer4" in name or "backbone.7" in name:
            p.requires_grad = True

    # 2. Freeze Mamba Backbone
    for p in encoder.parameters(): p.requires_grad = False

    # Unfreeze LoRA & Projections
    for name, p in encoder.named_parameters():
        if "lora" in name or "project_in" in name:
            p.requires_grad = True

    # 3. Summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f" Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


class CTCDecoder:
    """
    Decodes raw model outputs (logits or log-probs) into text strings.
    Handles greedy decoding and CTC blank removal + collapsing repeats.
    """

    def __init__(self, vocab: str, blank_idx: int = 0):
        self.vocab = vocab
        self.blank_idx = blank_idx  # usually 0

    def _to_batch_time_vocab(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is [B, T, V]. Accepts [B,T,V] or [T,B,V].
        """
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)

        if tensor.dim() != 3:
            raise ValueError("Expected 3D tensor [B,T,V] or [T,B,V]")

        # If shape is [T,B,V] -> permute to [B,T,V]
        if tensor.shape[0] != tensor.shape[1] and tensor.shape[2] == len(self.vocab) + 1:
            # ambiguous check but common case: if first dim equals T and second equals B
            # if shape[0] == tensor.size(1) it's likely [B,T,V] already; otherwise permute
            # safer check: if tensor.shape[0] < tensor.shape[1] and tensor.shape[2] == len(vocab)+1 -> assume [T,B,V]
            # We'll detect using typical relative sizes: if first dim is smaller than second -> probably T < B -> permute.
            if tensor.shape[0] < tensor.shape[1]:
                tensor = tensor.permute(1, 0, 2).contiguous()
        return tensor

    def decode_greedy(self, logits: torch.Tensor):
        """
        Greedy decode.
        Args:
            logits: [B,T,V]
        Returns:
            List[str] decoded strings
        """
        x = logits.detach().cpu()
        
        # Ensure 3D
        if x.dim() != 3:
             raise ValueError("CTCDecoder expects [B, T, V] logits")

        # argmax across vocab dim -> [B, T]

        # argmax across vocab dim -> [B, T]
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
                    # repeated char from CTC; skip
                    prev = ci
                    continue
                # append corresponding char
                # ci is 1..V-1 if blank=0, so vocab index = ci-1
                vocab_idx = ci - 1
                if 0 <= vocab_idx < len(self.vocab):
                    chars.append(self.vocab[vocab_idx])
                prev = ci
            decoded.append("".join(chars))
        return decoded

    def decode_beam_search(self, logits: torch.Tensor, beam_width: int = 10):
        """
        Pure Python Beam Search Decoding for CTC.
        Args:
           logits: Tensor [B, T, V] (log-softmax probabilities)
           beam_width: int
        Returns:
           List[str]
        """
        # Ensure probs
        if not hasattr(self, 'log_softmax'):
            self.log_softmax = torch.nn.LogSoftmax(dim=2)
            
        # If input is raw logits, apply softmax? Standard CTC loss takes log_probs.
        # We assume input is logits, so we apply log_softmax conversion if values > 0 (heuristic)
        # But train.py sets outputs raw. We'll standardly apply log_softmax on features.
        
        batch_log_probs = torch.nn.functional.log_softmax(logits, dim=2).detach().cpu().numpy()
        decoded_batch = []
        
        for log_probs in batch_log_probs:
            # log_probs: [T, V]
            T, V = log_probs.shape
            
            # Beam: List of tuples (score, text_indices, last_char_idx)
            # Initialize with empty path
            # score = log probability (0.0 for log(1))
            beam = [(0.0, [], self.blank_idx)] 
            
            for t in range(T):
                next_beam = {}
                
                # Pruning: Only consider top K candidates at regular steps to speed up?
                # Simple version: Expand all.
                
                for score, seq, last_char in beam:
                    # Try extending with every possible character
                    # Optimization: Only take top K probs at this step
                    # To keep it fast in python
                    
                    # Sort current step probs
                    step_probs = log_probs[t]
                    top_k_indices = np.argsort(step_probs)[-beam_width:] 
                    
                    for char_idx in top_k_indices:
                        char_prob = step_probs[char_idx]
                        new_score = score + char_prob
                        
                        if char_idx == self.blank_idx:
                            # Blank: keep sequence same, update last char to blank
                            new_last = self.blank_idx
                            new_seq = tuple(seq) # Tuple for dict key
                        else:
                            if char_idx == last_char:
                                # Repeated char (AA -> A)
                                new_last = char_idx
                                new_seq = tuple(seq)
                            else:
                                # New char
                                new_last = char_idx
                                new_seq = tuple(seq + [char_idx])
                                
                        # Update best score for this (seq, last_char) state
                        key = (new_seq, new_last)
                        if key not in next_beam or new_score > next_beam[key]:
                            next_beam[key] = new_score
                            
                # Sort and keep top beam_width
                sorted_beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Reformat back to list for next iteration
                beam = []
                for (seq_tuple, last_char), sc in sorted_beam:
                    beam.append((sc, list(seq_tuple), last_char))
                    
            # Best path at end
            best_seq = beam[0][1]
            
            # Decode indices to string
            chars = []
            for idx in best_seq:
                if 0 <= idx - 1 < len(self.vocab):
                     chars.append(self.vocab[idx - 1])
            decoded_batch.append("".join(chars))
            
        return decoded_batch


class AverageMeter:
    """Standard average meter for logging scalar metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_metrics(predictions, targets, normalize_for_wer=True):
    """
    Computes CER and WER.
    jiwer expects: reference (ground truth), hypothesis (prediction).
    We accept lists of strings: predictions, targets (in that order) for compatibility with earlier code,
    but call jiwer with (targets, predictions).
    """
    # Ensure lists
    preds = ["" if p is None else str(p) for p in predictions]
    refs = ["" if t is None else str(t) for t in targets]

    if normalize_for_wer:
        # simple normalizations: strip and collapse spaces
        preds = [p.strip() for p in preds]
        refs = [r.strip() for r in refs]

    cer = jiwer.cer(refs, preds)
    wer = jiwer.wer(refs, preds)
    return cer, wer


def save_checkpoint(model, optimizer, epoch, loss, save_dir="checkpoints", filename="best_model.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    loss = ckpt.get('loss', None)
    print(f"Loaded checkpoint from epoch {start_epoch-1} (loss={loss})")
    return start_epoch, loss


def visualize_prediction(image_tensor, prediction, target=None, unnormalize=True, mean=None, std=None):
    """
    Display single image and prediction.
    image_tensor: torch.Tensor - [1,H,W] or [3,H,W] or [H,W]
    prediction/target: strings
    unnormalize: if True and mean/std provided, undo normalization
    """
    img = image_tensor.detach().cpu().numpy()

    # handle [3,H,W] or [1,H,W] or [H,W]
    if img.ndim == 3:
        # [C,H,W]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # H,W,3
        elif img.shape[0] == 1:
            img = img.squeeze(0)  # H,W
    elif img.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported image tensor shape for visualize_prediction")

    # unnormalize if requested
    if unnormalize and mean is not None and std is not None:
        # mean/std expected as lists of length 3
        mean = np.array(mean)
        std = np.array(std)
        if img.ndim == 3 and img.shape[2] == 3:
            img = (img * std) + mean
        else:
            # single channel
            img = (img * std[0]) + mean[0]

    # scale to 0-255 for plotting
    img = np.clip(img, 0.0, 1.0)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    plt.figure(figsize=(10, 2))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    title = f"Pred: {prediction}"
    if target is not None:
        title += f" | True: {target}"
    plt.title(title)
    plt.axis('off')
    plt.show()