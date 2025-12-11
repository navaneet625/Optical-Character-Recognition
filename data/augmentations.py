import albumentations as A
import cv2

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MultiplicativeNoise(multiplier=[0.9, 1.1], elementwise=True, p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.3, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
    ])
