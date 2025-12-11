import os
import random
import string
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
import shutil

def get_fonts():
    # Common font paths for Mac and Linux (Kaggle)
    search_paths = [
        "/Library/Fonts/*.ttf",
        "/System/Library/Fonts/*.ttf",
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.otf",
        "/usr/local/share/fonts/**/*.ttf"
    ]
    fonts = []
    for path in search_paths:
        fonts.extend(glob.glob(path, recursive=True))
    
    # Filter out weird fonts (symbol fonts etc if possible, but hard to know)
    if not fonts:
        print("Warning: No TTF fonts found. Using default PIL font (very basic).")
    else:
        print(f"Found {len(fonts)} fonts available.")
    return fonts

def create_advanced_dummy_data(data_dir="data", num_samples=1000):
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if os.path.exists(images_dir): shutil.rmtree(images_dir)
    if os.path.exists(labels_dir): shutil.rmtree(labels_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    available_fonts = get_fonts()
    
    data = []
    possible_chars = string.ascii_letters + string.digits
    
    print(f"Generating {num_samples} ADVANCED samples...")
    
    for i in range(num_samples):
        # 1. Canvas
        bg_color = random.randint(200, 255) # Light background
        img = Image.new('RGB', (320, 64), color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)
        
        # 2. Text
        text_len = random.randint(3, 10)
        text = "".join(random.choices(possible_chars, k=text_len))
        
        # Font Selection
        font_size = random.randint(32, 48)
        font = None
        if available_fonts:
            try:
                font_path = random.choice(available_fonts)
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
            
        # Draw Text
        # Random Position
        try:
             # getbbox available in newer Pillow, otherwise getsize
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text)
                w, h = bbox[2], bbox[3]
            else:
                w, h = font.getsize(text)
        except:
             w, h = 100, 32 # fallback
             
        x = random.randint(10, max(11, 320 - w - 10))
        y = random.randint(5, max(6, 64 - h - 5))
        
        text_color = random.randint(0, 100)
        draw.text((x, y), text, fill=(text_color, text_color, text_color), font=font)
        
        # 3. Distortions
        # Rotation
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=False, fillcolor=bg_color)
        
        # Blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
            
        # Noise
        img_np = np.array(img)
        noise = np.random.normal(0, 15, img_np.shape).astype(np.uint8)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        # Resize to Target (320x32 for dataset)
        img = img.resize((320, 32), Image.Resampling.BILINEAR)
        
        # Save
        filename = f"{text}_{i}.png"
        path = os.path.join(images_dir, filename)
        img.save(path)
        data.append({"filename": filename, "label": text})
        
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(data)
    csv_path = os.path.join(labels_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    
    # Also save TXT for load_data compat
    txt_path = os.path.join(data_dir, "labels.txt")
    df.to_csv(txt_path, sep=" ", header=False, index=False)
    
    print(f"Generated {num_samples} samples.")

if __name__ == "__main__":
    # Default to 100 for local test, user changes for heavy run
    create_advanced_dummy_data(num_samples=100)
