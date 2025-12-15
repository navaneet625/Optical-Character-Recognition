import torch
import cv2
import numpy as np
import os
import re

from configs.config import Config
from models.ocr_model import MambaOCR
from utils import CTCDecoder

class OCRPredictor:
    def __init__(self, checkpoint_path=None):
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing LPR Engine on {self.device}")

        # 1. Initialize Model
        self.model = MambaOCR(
            vocab_size=len(self.cfg.vocab) + 1,
            cnn_out=self.cfg.cnn_out,
            n_layers=self.cfg.mamba_layers,
            adapter_dim=self.cfg.adapter_dim,
            lora_rank=self.cfg.lora_rank
        ).to(self.device)

        # 2. Load Weights 
        if checkpoint_path is None:
            options = [
                "checkpoints_finetuned/finetuned_round2_best.pth",
                "checkpoints_finetuned/finetuned_best.pth",
                self.cfg.best_model_path
            ]
            for path in options:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f" No valid checkpoint found! Checked: {options}")

        print(f" Loading weights: {os.path.basename(checkpoint_path)}")
        
        if not torch.cuda.is_available():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        else:
            ckpt = torch.load(checkpoint_path)

        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(clean_state_dict, strict=False)
        except Exception as e:
            print(f" Warning during load: {e}")

        self.model.eval()
        self.decoder = CTCDecoder(self.cfg.vocab)

    def _apply_regex(self, text):
        """Fixes common OCR typos (O->0, S->5) based on Indian Plate Format."""
        text = list(text)
        n = len(text)
        char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}
        num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}

        # AA (State)
        for i in range(min(2, n)):
            if text[i] in num_to_char: text[i] = num_to_char[text[i]]
        # 00 (District)
        for i in range(2, min(4, n)):
            if text[i] in char_to_num: text[i] = char_to_num[text[i]]
        # 0000 (Last 4 digits)
        if n >= 8:
            for i in range(n-4, n):
                if text[i] in char_to_num: text[i] = char_to_num[text[i]]
        return "".join(text)

    def _preprocess_batch(self, image_list):
        """Prepares a batch of images for the model (Resize -> Pad -> Normalize -> 3CH)."""
        batch_tensors = []
        # Pre-calc stats
        mean = np.array(self.cfg.mean).reshape(3, 1, 1)
        std = np.array(self.cfg.std).reshape(3, 1, 1)

        for img in image_list:
            h, w = img.shape
            scale = self.cfg.img_height / h
            new_w = min(int(w * scale), self.cfg.img_width)
            img = cv2.resize(img, (new_w, self.cfg.img_height))
            
            delta_w = self.cfg.img_width - new_w
            if delta_w > 0:
                img = cv2.copyMakeBorder(img, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=0)

            # Float & 3-Channel Stack
            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0) # [1,H,W] -> [3,H,W]
            
            # Normalize
            img = (img - mean) / std
            batch_tensors.append(torch.from_numpy(img).float())
        
        return torch.stack(batch_tensors).to(self.device)

    def predict(self, image_source, use_tta=True):
        """
        Main Inference Function.
        Args:
            image_source: Path to image (str) OR Numpy array (cv2 image).
            use_tta: Enable Test Time Augmentation (Sharpening).
        Returns:
            final_text: The cleaned, post-processed license plate string.
        """
        # 1. Read Image
        if isinstance(image_source, str):
            if not os.path.exists(image_source): return "ERR_FILE_NOT_FOUND"
            img_gray = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
        else:
            # Assume input is BGR or Gray numpy array
            if len(image_source.shape) == 3:
                img_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_source

        if img_gray is None: return "ERR_READ_IMAGE"

        # 2. Prepare Inputs (TTA or Single)
        images_to_process = [img_gray]
        
        if use_tta:
            # Add Sharpened version to fix '6' vs '8'
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(img_gray, -1, kernel)
            images_to_process.append(sharp)

        # 3. Model Inference
        batch_tensor = self._preprocess_batch(images_to_process)
        
        with torch.no_grad():
            logits = self.model(batch_tensor)
            # Average logits across TTA views (Noise cancelling)
            avg_logits = torch.mean(logits, dim=0, keepdim=True)
            
            # Try Beam Search first (Better accuracy)
            try:
                text = self.decoder.decode_beam_search(avg_logits, beam_width=5)[0]
            except:
                text = self.decoder.decode_greedy(avg_logits)[0]

        # 4. Regex Correction (Fix Typos)
        final_text = self._apply_regex(text)

        return final_text

if __name__ == "__main__":
    # Test on a real image
    TEST_IMAGE = "data/images/sample.jpg" 
    
    try:
        predictor = OCRPredictor() # Auto-loads best model
        
        if os.path.exists(TEST_IMAGE):
            print(f"   Processing: {TEST_IMAGE}")
            result = predictor.predict(TEST_IMAGE, use_tta=True)
            print(f"\n✅ LICENSE PLATE: {result}")
        else:
            print(f"⚠️ Test image not found at {TEST_IMAGE}. Please update path.")
            
    except Exception as e:
        print(f" Failed: {e}")