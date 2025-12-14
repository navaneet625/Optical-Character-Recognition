import albumentations as A
import cv2

def get_train_transforms():
    return A.Compose([
        # 1. Geometry
        # removed 'value=0' (defaults to black anyway)
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        
        # removed 'pad_mode'/'pad_val' (defaults to black/constant)
        A.Perspective(scale=(0.02, 0.05), p=0.3),

        # 2. Blur & Noise
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=1.0),
        ], p=0.5),

        # 3. Occlusions
        # Simplified parameters to avoid version conflicts
        A.CoarseDropout(max_holes=6, max_height=8, max_width=8, p=0.3),

        # 4. Lighting
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])