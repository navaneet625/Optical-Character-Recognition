import torch
import cv2
import numpy as np
from models.ocr_model import MambaOCR
from configs.config import Config

class OCRPredictor:
    def __init__(self, checkpoint_path):
        self.cfg = Config()
        self.device = torch.device(self.cfg.device)
        self.model = MambaOCR(vocab_size=len(self.cfg.vocab)+1,
                              cnn_out=self.cfg.cnn_out,
                              n_layers=self.cfg.mamba_layers).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Resize logic same as dataset
        h, w = img.shape
        new_w = int(w * (self.cfg.img_height / h))
        img = cv2.resize(img, (new_w, self.cfg.img_height))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        return img.to(self.device)

    def decode(self, preds):
        # Greedy decoding
        pred_indices = torch.argmax(preds, dim=2).detach().cpu().numpy() # [B, T]
        decoded_texts = []
        for sequence in pred_indices:
            text = []
            prev_char = -1
            for char_idx in sequence:
                if char_idx != self.cfg.blank_idx and char_idx != prev_char:
                    text.append(self.cfg.vocab[char_idx - 1])
                prev_char = char_idx
            decoded_texts.append("".join(text))
        return decoded_texts

    def predict(self, img_path):
        img_tensor = self.preprocess(img_path)
        with torch.no_grad():
            preds = self.model(img_tensor) # [B, T, C]
            preds = preds.softmax(2)
        
        text = self.decode(preds)
        return text[0]

    def export_onnx(self, output_path="ocr.onnx"):
        dummy_input = torch.randn(1, 1, 32, 320).to(self.device)
        torch.onnx.export(self.model, dummy_input, output_path, 
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {3: "width"}, "output": {1: "seq_len"}})
        print("Model exported to ONNX")

if __name__ == "__main__":
    predictor = OCRPredictor("checkpoints/mamba_ocr_ep49.pth")
    print(predictor.predict("data/test_image.png"))
    # predictor.export_onnx()