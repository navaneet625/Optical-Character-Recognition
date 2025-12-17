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
    Loads data from CSV or TXT. 
    Returns paths, labels, and weights.
    """
    image_paths = []
    labels = []
    weights = []

    # 1. Select Source File
    if hasattr(cfg, 'train_csv') and os.path.exists(cfg.train_csv):
        target_file = cfg.train_csv
        print(f" Loading Data from: {target_file}")
        df = pd.read_csv(target_file)
    elif hasattr(cfg, 'labels_file') and os.path.exists(cfg.labels_file):
        target_file = cfg.labels_file
        print(f" Loading Synthetic Data from: {target_file}")
        try:
            df = pd.read_csv(target_file, sep=" ", header=None, names=["filename", "label"])
        except:
             df = pd.read_csv(target_file)
    else:
        raise FileNotFoundError(" No valid labels file found in Config paths.")

    # Pre-processing: Drop NaNs
    df = df.dropna(subset=['label'])
    df = df[df['label'].astype(str).str.lower() != 'nan']
    
    # Handle Weights (Default to 1.0)
    if 'weight' not in df.columns:
        df['weight'] = 1.0
    
    print(f"Processing {len(df)} paths...")

    # Path Construction
    base_dir = cfg.images_dir if hasattr(cfg, 'images_dir') else os.path.join(cfg.data_dir, "images")
    
    filenames = df["filename"].astype(str).values
    raw_labels = df["label"].astype(str).values
    raw_weights = df["weight"].astype(float).values
    
    # Clean paths
    clean_filenames = [
        f[7:] if f.startswith("images/") else 
        f[12:] if f.startswith("data/images/") else 
        f[2:] if f.startswith("./") else f 
        for f in filenames
    ]
    
    # Construct Full Paths
    image_paths = [os.path.join(base_dir, f) for f in clean_filenames]
    labels = list(raw_labels)
    weights = list(raw_weights)

    # Sanity Check
    if len(image_paths) > 0 and not os.path.exists(image_paths[0]):
        print(f"WARNING: First file not found at: {image_paths[0]}")
        print("Check your 'config.py' paths!")
    
    print(f"Loaded {len(image_paths)} samples.")
    return image_paths, labels, weights

class OCRDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        
        # Load Data (Paths, Labels, Weights)
        self.image_paths, self.labels, self.weights = load_data(cfg)
        
        # Map Chars
        self.char2idx = {c: i+1 for i, c in enumerate(cfg.vocab)}
        
        # Augmentations
        self.aug = get_train_transforms() if is_train else None

        # Caching: Disable for large datasets (>20k) to save RAM
        self.use_cache = len(self.image_paths) < 20000
        
        if self.use_cache:
            print(f"Caching {len(self.image_paths)} images in RAM...")
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
            return np.zeros((self.cfg.img_height, self.cfg.img_width), dtype=np.uint8)

    def encode(self, text):
        text = str(text).upper()
        text = re.sub(r'[^0-9A-Z]', '', text)
        cleaned = [self.char2idx[c] for c in text if c in self.char2idx]
        if not cleaned: cleaned = [self.char2idx[self.cfg.vocab[0]]] 
        return torch.tensor(cleaned, dtype=torch.long)

    def resize_and_pad(self, img):
        h, w = img.shape
        target_h, target_w = self.cfg.img_height, self.cfg.img_width
        
        scale = target_h / h
        new_w = min(int(w * scale), target_w)
        img = cv2.resize(img, (new_w, target_h))
        
        delta_w = target_w - new_w
        if delta_w > 0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=0)
            
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_cache:
            img = self.cached_images[idx]
        else:
            img = self.load_image_file(self.image_paths[idx])

        if self.is_train and self.aug:
            try: img = self.aug(image=img)["image"]
            except: pass

        img = self.resize_and_pad(img)

        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)
        
        mean = np.array(self.cfg.mean)[:, None, None]
        std = np.array(self.cfg.std)[:, None, None]
        img = (img - mean) / std
        
        label_text = self.labels[idx]
        target = self.encode(label_text)
        target_len = torch.tensor(len(target), dtype=torch.long)
        
        # Return Weight for Loss Function
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)

        return torch.from_numpy(img).float(), target, target_len, weight