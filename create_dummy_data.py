import cv2
import numpy as np
import os

def create_dummy_data(base_dir="data", num_samples=10):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
        
    print(f"Generating {num_samples} dummy images in {images_dir}...")
    
    vocab = "0123456789abcdefghijklmnopqrstuvwxyz"
    
    import pandas as pd
    
    data = []
    for i in range(num_samples):
        # Create a white image
        img = np.ones((32, 128), dtype=np.uint8) * 255
        
        # Random text
        text_len = np.random.randint(3, 8)
        text = "".join([vocab[np.random.randint(0, len(vocab))] for _ in range(text_len)])
        
        # Put text on image
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
        
        # Save
        filename = f"{text}_{i}.png"
        path = os.path.join(images_dir, filename)
        cv2.imwrite(path, img)
        print(f"Saved {path}")
        
        data.append({"filename": filename, "label": text})
        
    # Save CSV
    csv_path = os.path.join(labels_dir, "labels.csv")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Saved labels.csv to {csv_path} with {len(df)} entries.")

if __name__ == "__main__":
    create_dummy_data()
