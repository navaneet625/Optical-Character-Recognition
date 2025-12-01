import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import pandas as pd
from .augmentations import get_train_transforms

def load_data(cfg):
    """
    Automatically discovers images and labels.
    Uses paths from config: cfg.images_dir and cfg.labels_file
    """
    image_paths = []
    labels = []
    
    # 1. Check for labels.csv
    csv_path = cfg.labels_file
    if os.path.exists(csv_path):
        print(f"Found labels.csv at {csv_path}")
        df = pd.read_csv(csv_path)
        # Expect columns: 'filename', 'label'
        if 'filename' not in df.columns or 'label' not in df.columns:
            print("Warning: labels.csv missing 'filename' or 'label' headers. Using first two columns.")
            df.columns = ['filename', 'label'] + list(df.columns[2:])
            
        for _, row in df.iterrows():
            # Look for image in images_dir
            img_p = os.path.join(cfg.images_dir, str(row['filename']))
            if os.path.exists(img_p):
                image_paths.append(img_p)
                labels.append(str(row['label']))
            else:
                # Fallback: Check if filename is absolute or relative to data_dir
                img_p_alt = os.path.join(cfg.data_dir, str(row['filename']))
                if os.path.exists(img_p_alt):
                    image_paths.append(img_p_alt)
                    labels.append(str(row['label']))
    else:
        print(f"No labels.csv found at {csv_path}. Using filename as label from {cfg.images_dir}.")
        # 2. Glob images from images_dir
        if not os.path.exists(cfg.images_dir):
             print(f"Warning: Images directory {cfg.images_dir} does not exist. Checking data_dir {cfg.data_dir}...")
             search_dir = cfg.data_dir
        else:
             search_dir = cfg.images_dir
             
        # Recursive glob
        files = glob.glob(os.path.join(search_dir, "**", "*.*"), recursive=True)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                image_paths.append(f)
                basename = os.path.basename(f)
                label = os.path.splitext(basename)[0]
                labels.append(label)
                
    print(f"Loaded {len(image_paths)} images.")
    return image_paths, labels

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, config, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.cfg = config
        # Create char map (1-based index, 0 is reserved for CTC Blank)
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.cfg.vocab)}
        self.is_train = is_train
        self.aug = get_train_transforms()

    def encode_text(self, text):
        # Convert string to list of indices
        # Ignores characters not in vocab
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def resize_aspect_ratio(self, img):
        """
        Resizes height to 32, maintains aspect ratio for width.
        Does NOT pad here. We let collate_fn do the padding.
        """
        h, w = img.shape
        # Calculate new width to maintain aspect ratio
        new_w = int(w * (self.cfg.img_height / h))
        
        # Limit maximum width to avoid OOM on very long images
        if new_w > self.cfg.img_width:
            new_w = self.cfg.img_width
            
        # Resize
        img = cv2.resize(img, (new_w, self.cfg.img_height))
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Safety Check: File exists?
        if not os.path.exists(img_path):
            print(f"Warning: File not found {img_path}")
            # Return a dummy black image to prevent crash
            return torch.zeros(1, 32, 32), torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long)

        # 2. Read Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Safety Check: Corrupt image?
        if img is None:
            print(f"Warning: Could not read {img_path}")
            return torch.zeros(1, 32, 32), torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long)
        
        label = str(self.labels[idx]) # Ensure label is string

        # 3. Augment (Train only)
        if self.is_train:
            try:
                # Albumentations expects RGB usually, but works on Gray if config allowed
                # Safer to wrap in try-catch for edge cases
                res = self.aug(image=img)
                img = res['image']
            except Exception as e:
                pass

        # 4. Resize (Height=32, Width=Variable)
        img = self.resize_aspect_ratio(img)

        # 5. Normalize & Tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0) # [1, H, W]
        
        # 6. Encode Label
        encoded_label = torch.tensor(self.encode_text(label), dtype=torch.long)
        label_len = torch.tensor([len(encoded_label)], dtype=torch.long)
        
        return img, encoded_label, label_len