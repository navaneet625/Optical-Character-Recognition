import torch
import cv2
import numpy as np
import os
from models.ocr_model import MambaOCR
from configs.config import Config
from utils import CTCDecoder

class MambaPredictor:
    def __init__(self, checkpoint_path="checkpoints/best_mamba_ocr.pth"):
        self.cfg = Config()
        self.device = torch.device(self.cfg.device)

        print(f"🍌 Loading NanoMamba OCR from {checkpoint_path}...")

        self.model = MambaOCR(
            vocab_size=len(self.cfg.vocab)+1,
            cnn_out=self.cfg.cnn_out,
            n_layers=self.cfg.mamba_layers,
            adapter_dim=self.cfg.adapter_dim,
            lora_rank=self.cfg.lora_rank
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Handle DataParallel (module.) prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        # Initialize Decoder
        self.decoder = CTCDecoder(self.cfg.vocab)

    def preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found or corrupt: {image_path}")

        h, w = img.shape
        new_w = int(w * (self.cfg.img_height / h))
        img = cv2.resize(img, (new_w, self.cfg.img_height))

        img = img.astype(np.float32) / 255.0
        
        # Stack to 3 channels (for ResNet)
        img = np.stack([img, img, img], axis=0) # [3, H, W]
        
        # Normalize (ImageNet stats)
        mean = np.array(self.cfg.mean)[:, None, None]
        std = np.array(self.cfg.std)[:, None, None]
        img = (img - mean) / std

        tensor = torch.from_numpy(img).unsqueeze(0).float() # [1, 3, H, W]
        return tensor.to(self.device)

    def predict(self, image_path, use_beam=True):
        tensor = self.preprocess(image_path)

        with torch.no_grad():
            logits = self.model(tensor)

        if use_beam:
            # Returns list of strings
            texts = self.decoder.decode_beam_search(logits, beam_width=10)
            text = texts[0]
        else:
            texts = self.decoder.decode_greedy(logits)
            text = texts[0]

        return text

if __name__ == "__main__":
    predictor = MambaPredictor()

    test_img = "data/images/0mPPn9CS_72.png"

    if os.path.exists(test_img):
        print(f"Processing {test_img}...")
        try:
            result = predictor.predict(test_img, use_beam=True)
            print(f"\n Image: {test_img}")
            print(f"✨ Prediction: {result}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Image not found: {test_img}")
        print("Please edit 'test_img' in the script to point to a valid image.")