import albumentations as A
import cv2

def get_train_transforms():
    return A.Compose([
        # 1. Geometry (Rotation & Shear)
        # cval=255 ensures we pad with WHITE (background), not BLACK (ink)
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
        
        # Shear mimics handwriting slant
        A.Affine(shear={"x": (-10, 10)}, cval=255, p=0.3), 
        
        # Perspective (Tilt) - Pad with white (255)
        A.Perspective(scale=(0.02, 0.05), pad_mode=cv2.BORDER_CONSTANT, pad_val=255, p=0.3),
        
        # 2. Blur (Simulate bad focus)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        
        # 3. Noise (Simulate camera grain)
        # Note: If 'var_limit' throws a warning, update albumentations: pip install -U albumentations
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # 4. Lighting
        A.RandomBrightnessContrast(p=0.4),
        
        # REMOVED A.Normalize() because dataset.py handles the /255.0 scaling
    ])