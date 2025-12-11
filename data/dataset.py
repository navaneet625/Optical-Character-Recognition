import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
from .augmentations import get_train_transforms

def load_data(cfg):
    """
    Loads data from labels.txt (preferred) or labels.csv.
    Returns: List[Path], List[str]
    """
    image_paths = []
    labels = []

    # Check for direct file path logic
    txt_path = os.path.join(cfg.data_dir, "labels.txt")
    csv_path = cfg.labels_file

    df = None
    if os.path.exists(txt_path):
        print(f"Loading {txt_path}...")
        df = pd.read_csv(txt_path, sep=" ", header=None, names=["filename", "label"])
    elif os.path.exists(csv_path):
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        # Fallback for local dummy data if paths are messy
        local_csv = "data/labels/labels.csv"
        if os.path.exists(local_csv):
             print(f"Loading fallback {local_csv}...")
             df = pd.read_csv(local_csv)
    
    if df is None:
        raise FileNotFoundError("Could not find labels.txt or labels.csv")

    # Construct Paths
    count = 0 
    
    # Speed Optimization: Bulk Check
    # This avoids 10,000+ system calls to os.path.exists
    try:
        images_set = set(os.listdir(cfg.images_dir))
    except FileNotFoundError:
        images_set = set()
        
    for _, row in df.iterrows():
        fname = str(row["filename"])
        label = str(row["label"])
        
        # 1. Check Configured Dir (Fast)
        if fname in images_set:
            image_paths.append(os.path.join(cfg.images_dir, fname))
            labels.append(label)
            count += 1
            continue
            
        # 2. Check Fallback Local Dir (Slow fallback)
        fallback_path = os.path.join("data/images", fname)
        if os.path.exists(fallback_path):
             image_paths.append(fallback_path)
             labels.append(label)
             count += 1
            
    print(f"Loaded {count} valid samples.")
    return image_paths, labels


class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, cfg, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.cfg = cfg
        self.is_train = is_train

        # Vocab Mapping
        self.char2idx = {c: i+1 for i, c in enumerate(cfg.vocab)}
        
        # Augmentations
        self.aug = get_train_transforms() if is_train else None

        # Caching
        print(f"Caching {len(image_paths)} images...")
        self.cached_images = []
        for p in tqdm(image_paths):
            try:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None: raise ValueError
                self.cached_images.append(img)
            except:
                self.cached_images.append(np.zeros((cfg.img_height, cfg.img_width), dtype=np.uint8))

    def encode(self, text):
        # Filter valid chars
        cleaned = [self.char2idx[c] for c in text if c in self.char2idx]
        if not cleaned: cleaned = [self.char2idx[self.cfg.vocab[0]]] 
        return torch.tensor(cleaned, dtype=torch.long)

    def resize(self, img):
        h, w = img.shape
        # Aspect Ratio Resize
        new_w = min(int(w * (self.cfg.img_height / h)), self.cfg.img_width)
        return cv2.resize(img, (new_w, self.cfg.img_height))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        label = self.labels[idx]

        # 1. Augment
        if self.is_train and self.aug:
            try: img = self.aug(image=img)["image"]
            except: pass

        # 2. Resize
        img = self.resize(img)

        # 3. Normalize & Tensor (Float32)
        img = img.astype(np.float32) / 255.0
        
        # 4. Convert Grayscale -> RGB (for ResNet)
        img = np.stack([img, img, img], axis=0) # [3, H, W]

        # 5. ImageNet Norm
        mean = np.array(self.cfg.mean)[:, None, None]
        std = np.array(self.cfg.std)[:, None, None]
        img = (img - mean) / std
        
        img = torch.from_numpy(img).float()
        
        # 6. Label
        target = self.encode(label)
        target_len = torch.tensor(len(target), dtype=torch.long)

        return img, target, target_len
