import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=5, border_mode=0, p=0.4),
        A.Perspective(scale=(0.03, 0.05), p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(var_limit=(5, 20), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        A.Normalize(mean=0.5, std=0.5),
    ])

