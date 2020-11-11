from torch.utils.data import DataLoader
from transforms import get_image_transform
from .car_dataset import CarDataset
import torch


def get_dataloader(shuffle=True):
    # config
    img_folder_path = "./data/training_data"
    csv_path = "./data/training_labels.csv"
    batch_size = 16
    num_worker = 4
    split = 0.2
    
    dataset_transform = get_image_transform()
    
    myDataset = CarDataset(
        img_folder_path=img_folder_path,
        csv_path=csv_path,
        transform=dataset_transform['train'],
    )
    
    # Creating dataset for training and validation splits:
    valid_size = int(len(myDataset) * split)
    train_size = int(len(myDataset)) - valid_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        myDataset, [train_size, valid_size]
    )
    
    validation_dataset.transformations = dataset_transform['val']

    # Define data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
    )
    
    dataset_loaders = {"train": train_loader, "val": validation_loader}
    dataset_sizes = {"train": train_size, "val": valid_size}
    dataset_map = myDataset.unique_classes

    return dataset_loaders, dataset_sizes, dataset_map


def test():
    #non
    a = 0

if __name__ == '__main__':
    test()
