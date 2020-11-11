import torchvision.transforms as trans
from . import autoaugment

# .ImageNetPolicy as ImageNetPolicy


def get_image_transform():
    # ---------------------------------------------------
    # Create train/valid transforms.
    # Return transform (Dict): 'train' for training; 'val' for validation and testing.
    # ---------------------------------------------------
    train_transform = trans.Compose(
        [
            trans.Resize(256),
            trans.RandomResizedCrop(224),
            trans.RandomHorizontalFlip(),
            autoaugment.ImageNetPolicy(),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = trans.Compose(
        [
            trans.Resize((224, 224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_transform = {"train": train_transform, "val": test_transform}

    return dataset_transform
