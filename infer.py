import torch
import cv2
import numpy as np
import os
from models.ocr_model import MambaOCR
from configs.config import Config
from utils import CTCDecoder
import numpy as np

class MambaPredictor:
    def __init__(self, checkpoint_path="/Users/navneetsingh/Downloads/checkpoints/best_mamba_ocr.pth"):
        self.cfg = Config()
        self.device = torch.device(self.cfg.device)

        print(f"🍌 Loading NanoMamba OCR from {checkpoint_path}...")

        # Initialize Model
        self.model = MambaOCR(
            vocab_size=len(self.cfg.vocab) + 1,
            cnn_out=self.cfg.cnn_out,
            n_layers=self.cfg.mamba_layers,
            adapter_dim=self.cfg.adapter_dim,
            lora_rank=self.cfg.lora_rank
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        # Load Weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            new_state_dict[key] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        self.decoder = CTCDecoder(self.cfg.vocab)

    def preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found or corrupt: {image_path}")

        h, w = img.shape
        target_h, target_w = self.cfg.img_height, self.cfg.img_width 
        
        scale = target_h / h
        new_w = int(w * scale)
        new_w = min(new_w, target_w)
        
        img = cv2.resize(img, (new_w, target_h))

        delta_w = target_w - new_w
        if delta_w > 0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=0)

        img = img.astype(np.float32) / 255.0

        img = np.stack([img, img, img], axis=0)
  
        mean = np.array(self.cfg.mean)[:, None, None]
        std = np.array(self.cfg.std)[:, None, None]
        img = (img - mean) / std

        tensor = torch.from_numpy(img).unsqueeze(0).float() 
        return tensor.to(self.device)

    def predict(self, image_path, use_beam=True):
        tensor = self.preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor) 

        if use_beam:
            texts = self.decoder.decode_beam_search(logits, beam_width=10)
            return texts[0]
        else:
            texts = self.decoder.decode_greedy(logits)
            return texts[0]

if __name__ == "__main__":
    dummy_path = "IIIT5K/test/6_4.png"

    # dummy_img = np.zeros((64, 200, 3), dtype=np.uint8) + 255
    # cv2.putText(dummy_img, "Test123", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # cv2.imwrite(dummy_path, dummy_img)

    predictor = MambaPredictor()
    
    print(f"Processing {dummy_path}...")
    result = predictor.predict(dummy_path)
    print(f"Pred: {result}")