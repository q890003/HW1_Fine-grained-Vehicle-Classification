from __future__ import print_function, division

from torch.utils.data.dataset import Dataset
import torchvision.transforms as trans

from PIL import Image
import pandas as pd
import numpy as np


class CarDataset(Dataset):
    def __init__(self, img_folder_path, csv_path, transform=None):
        """
        --------------------------------------------
        Initialize paths, transforms, and so on
        Args:
            img_dir (string): Directory with all the images.
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        --------------------------------------------
        """
        super(CarDataset, self).__init__()
        # Transforms
        self.transformations = transform

        # Load image path and annotations
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)  # labels = print(df.iloc[0].get(1))
        # First column contains the image paths
        self.img_ids = np.asarray(self.data_info.iloc[:, 0])
        # Load image path from the folder_path and pandas df
        self.img_list = [
            img_folder_path + "/" + str(img_id).zfill(6) + ".jpg"
            for img_id in self.img_ids
        ]
        # Second column is the labels (labels is transformed from str to numbers.)
        self.img_labels, self.unique_classes = pd.factorize(
            self.data_info["label"].values.tolist(), sort=True
        )
        # Calculate len
        self.data_len = len(self.data_info.index)
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------

        # Get image filename from the img_list
        single_image_name = self.img_list[index]

        # Open image
        img_as_img = Image.open(single_image_name).convert("RGB")

        # Transform image to tensor
        img_as_tensor = self.transformations(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.img_labels[index]

        return img_as_tensor, single_image_label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return self.data_len


def test():
    train_transform = trans.Compose(
        [
            trans.RandomResizedCrop(224),
            trans.ToTensor(),
        ]
    )

    dataset = CarDataset(
        img_folder_path="./data/training_data",
        csv_path="./data/training_labels.csv",
        transform=train_transform,
    )
    
    print(len(dataset))
          
if __name__ == '__main__':
    test()
    
