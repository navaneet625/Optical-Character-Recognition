import os
import torch
import torch.nn as nn
import jiwer
import numpy as np

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

def apply_freeze_strategy(model, cfg):
    print("Applying Freeze Strategy...")
    
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

    def decode_beam_search(self, logits: torch.Tensor, beam_width: int = 10):
        """
        Pure Python Beam Search Decoding for CTC.
        Args:
           logits: Tensor [B, T, V] (raw logits)
           beam_width: int
        Returns:
           List[str]
        """
        # Apply log_softmax to get log probabilities
        batch_log_probs = torch.nn.functional.log_softmax(logits, dim=2).detach().cpu().numpy()
        decoded_batch = []
        
        for log_probs in batch_log_probs:
            # log_probs: [T, V]
            T, V = log_probs.shape
            
            # Beam: List of tuples (score, text_indices, last_char_idx)
            # score = log probability (0.0 for log(1))
            beam = [(0.0, [], self.blank_idx)] 
            
            for t in range(T):
                next_beam = {}
                
                # Sort beam by score to prioritize high prob paths
                # Optimization: Only expand the current top beam_width candidates
                beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]

                for score, seq, last_char in beam:
                    # Optimization: Only take top K probs at this specific time step t                    
                    step_probs = log_probs[t]
                    # Get indices of top K probabilities
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
                            
                # Sort by score and keep top beam_width for next iteration
                sorted_beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Reformat back to list for next iteration
                beam = []
                for (seq_tuple, last_char), sc in sorted_beam:
                    beam.append((sc, list(seq_tuple), last_char))
                    
            # Best path at end of time T
            best_seq = beam[0][1]
            
            # Decode indices to string
            chars = []
            for idx in best_seq:
                if 0 <= idx - 1 < len(self.vocab):
                     chars.append(self.vocab[idx - 1])
            decoded_batch.append("".join(chars))
            
        return decoded_batch

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
    print(f"Checkpoint saved: {path}")