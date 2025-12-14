import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import pandas as pd
import re
from tqdm import tqdm
from .augmentations import get_train_transforms

def load_data(cfg):
    """
    Robust data loader. 
    1. Checks for config paths.
    2. Filters missing images.
    """
    image_paths = []
    labels = []

    # Decide which file to load based on Config (Synth vs Real)
    # Priority: train_csv (Real) -> labels_file (Synth)
    if hasattr(cfg, 'train_csv') and os.path.exists(cfg.train_csv):
        target_file = cfg.train_csv
        print(f"Loading Real Data from: {target_file}")
        df = pd.read_csv(target_file)
    elif hasattr(cfg, 'labels_file') and os.path.exists(cfg.labels_file):
        target_file = cfg.labels_file
        print(f"Loading Synthetic Data from: {target_file}")
        # Assuming space separated txt for synthetic
        try:
            df = pd.read_csv(target_file, sep=" ", header=None, names=["filename", "label"])
        except:
             # Fallback if it's actually a CSV
             df = pd.read_csv(target_file)
    else:
        raise FileNotFoundError(" No valid labels file found in Config paths.")

    # Iterate and Validate
    valid_count = 0
    
    # Check if 'dir_path' column exists (for merged datasets)
    has_dir_col = 'dir_path' in df.columns
    
    for _, row in df.iterrows():
        fname = str(row["filename"])
        label = str(row["label"])
        
        # Skip bad labels
        if label.lower() == 'nan': continue

        # # Resolve Path
        # if has_dir_col and pd.notna(row['dir_path']):
        #      full_path = os.path.join(row['dir_path'], fname)
        # elif hasattr(cfg, 'images_dir'):
        #      full_path = os.path.join(cfg.images_dir, fname)
        # else:
        #      full_path = os.path.join(cfg.data_dir, "images", fname)
             
        # # Check existence (Optional: disable for massive speedup if confident)
        # if os.path.exists(full_path):
        #     image_paths.append(full_path)
        #     labels.append(label)
        #     valid_count += 1

        cleaned_fname = fname
        for prefix in ['images/', 'data/images/', './images/']:
            if cleaned_fname.startswith(prefix):
                cleaned_fname = cleaned_fname[len(prefix):]
                break

        # Resolve Path using the cleaned filename
        if has_dir_col and pd.notna(row['dir_path']):
             full_path = os.path.join(row['dir_path'], cleaned_fname)
        elif hasattr(cfg, 'images_dir'):
             full_path = os.path.join(cfg.images_dir, cleaned_fname)
        else:
             # Fallback if images_dir is not set
             full_path = os.path.join(cfg.data_dir, "images", cleaned_fname)
             
        # Check existence 
        if os.path.exists(full_path):
            image_paths.append(full_path)
            labels.append(label)
            valid_count += 1
            
    print(f"Loaded {valid_count} valid samples.")
    return image_paths, labels

class OCRDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        
        # Load Data
        self.image_paths, self.labels = load_data(cfg)
        
        # Map Chars (0-9A-Z)
        self.char2idx = {c: i+1 for i, c in enumerate(cfg.vocab)}
        
        # Augmentations
        self.aug = get_train_transforms() if is_train else None

        # Caching Strategy
        # Only cache if dataset < 20k images to avoid RAM explosion
        self.use_cache = len(self.image_paths) < 20000
        
        if self.use_cache:
            print(f"⚡ Caching {len(self.image_paths)} images in RAM...")
            self.cached_images = []
            for p in tqdm(self.image_paths):
                img = self.load_image_file(p)
                self.cached_images.append(img)
        else:
            self.cached_images = None

    def load_image_file(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError
            return img
        except:
            # Return Black Image on failure
            return np.zeros((self.cfg.img_height, self.cfg.img_width), dtype=np.uint8)

    def encode(self, text):
        """
        STRICT UPPERCASE ENFORCEMENT
        """
        # 1. Force Upper
        text = str(text).upper()
        
        # 2. Filter using Vocab (Keep only 0-9A-Z)
        text = re.sub(r'[^0-9A-Z]', '', text)
        
        # 3. Map to Index
        cleaned = [self.char2idx[c] for c in text if c in self.char2idx]
        
        # 4. Handle Empty
        if not cleaned: cleaned = [self.char2idx[self.cfg.vocab[0]]] 
        return torch.tensor(cleaned, dtype=torch.long)

    def resize_and_pad(self, img):
        """
        Resizes to height=32, preserves aspect ratio, then PADS width to 320.
        Essential for Batch Processing.
        """
        h, w = img.shape
        target_h, target_w = self.cfg.img_height, self.cfg.img_width # 32, 320
        
        # Scale to fixed height
        scale = target_h / h
        new_w = min(int(w * scale), target_w)
        img = cv2.resize(img, (new_w, target_h))
        
        # Pad Right side with Black (0) to match target_w
        delta_w = target_w - new_w
        if delta_w > 0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=0)
            
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Get Image
        if self.use_cache:
            img = self.cached_images[idx]
        else:
            img = self.load_image_file(self.image_paths[idx])

        # 2. Augment
        if self.is_train and self.aug:
            try: img = self.aug(image=img)["image"]
            except: pass

        # 3. Resize & Pad
        img = self.resize_and_pad(img)

        # 4. Normalize & Tensor
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0) # [1,H,W] -> [3,H,W]
        
        mean = np.array(self.cfg.mean)[:, None, None]
        std = np.array(self.cfg.std)[:, None, None]
        img = (img - mean) / std
        
        # 5. Label
        label_text = self.labels[idx]
        target = self.encode(label_text)
        target_len = torch.tensor(len(target), dtype=torch.long)

        return torch.from_numpy(img).float(), target, target_len