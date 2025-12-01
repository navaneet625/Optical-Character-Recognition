import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=5, p=0.5),
        A.GaussianBlur(p=0.2),
        A.MultiplicativeNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
    ])
