import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
from .augmentations import get_train_transforms

def load_data(cfg):
    # (Keep your existing load_data function exactly as is)
    image_paths = []
    labels = []

    csv_path = cfg.labels_file
    if os.path.exists(csv_path):
        print(f"Found labels.csv at {csv_path}")
        df = pd.read_csv(csv_path)
        if 'filename' not in df.columns or 'label' not in df.columns:
            df.columns = ['filename', 'label'] + list(df.columns[2:])

        for _, row in df.iterrows():
            img_p = os.path.join(cfg.images_dir, str(row['filename']))
            if os.path.exists(img_p):
                image_paths.append(img_p)
                labels.append(str(row['label']))
    else:
        # Glob fallback (Keep your existing glob logic)
        search_dir = cfg.images_dir if os.path.exists(cfg.images_dir) else cfg.data_dir
        files = glob.glob(os.path.join(search_dir, "**", "*.*"), recursive=True)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_paths.append(f)
                labels.append(os.path.splitext(os.path.basename(f))[0])

    print(f"Loaded {len(image_paths)} images.")
    return image_paths, labels

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, config, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.cfg = config
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.cfg.vocab)}
        self.is_train = is_train
        self.aug = get_train_transforms()

        # --- NEW: RAM CACHING LOGIC ---
        print(f"⚡ Caching {len(image_paths)} images to RAM...")
        self.cached_images = []
        for path in tqdm(image_paths):
            # Read in Grayscale as per your logic
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback blank image
                img = np.zeros((config.img_height, config.img_width), dtype=np.uint8)
            self.cached_images.append(img)
        print(" Cache Complete. Disk I/O eliminated.")
        # ------------------------------

    def encode_text(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def resize_aspect_ratio(self, img):
        h, w = img.shape
        new_w = int(w * (self.cfg.img_height / h))
        if new_w > self.cfg.img_width:
            new_w = self.cfg.img_width
        img = cv2.resize(img, (new_w, self.cfg.img_height))
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Get from RAM instead of Disk
        img = self.cached_images[idx]
        label = str(self.labels[idx])

        # 2. Augment
        if self.is_train:
            try:
                res = self.aug(image=img)
                img = res['image']
            except:
                pass

        # 3. Resize
        img = self.resize_aspect_ratio(img)

        # 4. Normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        # 5. Encode
        encoded_label = torch.tensor(self.encode_text(label), dtype=torch.long)
        label_len = torch.tensor([len(encoded_label)], dtype=torch.long)

        return img, encoded_label, label_len