import albumentations as A
import cv2

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.3),
        A.GaussianBlur(p=0.2),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])