import torch
import cv2
import numpy as np
import os
from models.ocr_model import MambaOCR
from configs.config import Config
from fast_ctc_decode import beam_search

class MambaPredictor:
    def __init__(self, checkpoint_path="checkpoints/best_mamba_ocr.pth"):
        self.cfg = Config()
        self.device = torch.device(self.cfg.device)
        
        print(f"🍌 Loading NanoMamba OCR from {checkpoint_path}...")
        
        # 1. Initialize Architecture
        self.model = MambaOCR(
            vocab_size=len(self.cfg.vocab)+1,
            cnn_out=self.cfg.cnn_out,
            n_layers=self.cfg.mamba_layers
        ).to(self.device)
        
        # 2. Load Weights (Robustly)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # 3. Prep Beam Search
        self.full_vocab = "-" + self.cfg.vocab
        
    def preprocess(self, image_path):
        # Read
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not found or corrupt.")
            
        # Resize (Height 32, Width Dynamic)
        h, w = img.shape
        new_w = int(w * (self.cfg.img_height / h))
        img = cv2.resize(img, (new_w, self.cfg.img_height))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Tensor [1, 1, H, W]
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path, use_beam=True):
            tensor = self.preprocess(image_path)
            
            with torch.no_grad():
                logits = self.model(tensor) # [1, Seq, Vocab]
                probs = torch.softmax(logits, dim=2).cpu().numpy()[0]
            
            if use_beam:
                # Beam Search returns (text, path_indices)
                beam_result = beam_search(probs, self.full_vocab, beam_size=10)
                
                # --- FIX: Extract text from tuple ---
                if isinstance(beam_result, tuple) or isinstance(beam_result, list):
                    text = beam_result[0]
                else:
                    text = beam_result
            else:
                # Greedy Search (Simple argmax)
                indices = np.argmax(probs, axis=1)
                text = ""
                for idx in indices:
                    if idx != 0 and (not text or text[-1] != self.cfg.vocab[idx-1]):
                        text += self.cfg.vocab[idx-1]
                        
            return text

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Create Predictor
    predictor = MambaPredictor()
    
    # 2. Pick a random test image from your generated set
    test_img = "data/images/syn_000005.jpg" 
    
    # 3. Run
    if os.path.exists(test_img):
        result = predictor.predict(test_img)
        print(f"\n🖼️ Image: {test_img}")
        print(f"✨ Prediction: {result}")
    else:
        print("Please provide a valid image path.")