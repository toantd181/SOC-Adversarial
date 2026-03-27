import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

class GTSRBDataset(Dataset):
    def __init__(self, data_source: Union[str, pd.DataFrame], root_dir: str, transform: Optional[transforms.Compose] = None):
        if isinstance(data_source, str):
            self.annotations = pd.read_csv(data_source)
        else:
            self.annotations = data_source.reset_index(drop = True)

        self.root_dir = root_dir
        self.transform = transform

    def __len__ (self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        relative_path = self.annotations.iloc[index]['Path']
        y_label = int(self.annotations.iloc[index]['ClassId'])

        img_path = os.path.join(self.root_dir, relative_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, y_label
    
def get_data_loaders(data_dir: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]) 
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]) # Mean/Std của GTSRB
    ])

    full_train_csv_path = os.path.join(data_dir, 'Train.csv')
    full_df = pd.read_csv(full_train_csv_path)

    train_df, val_df = train_test_split(
        full_df,
        test_size = 0.2,
        random_state = 42,
        stratify=full_df['ClassId']
    )

    val_dataset = GTSRBDataset(
        data_source= val_df,
        root_dir=data_dir, 
        transform=val_transforms
    )

    train_dataset = GTSRBDataset(
        data_source=train_df,
        root_dir = data_dir,
        transform = train_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return train_loader, val_loader

def get_test_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])    
    ])

    test_dataset = GTSRBDataset(
        data_source=os.path.join(data_dir, 'Test.csv'),
        root_dir=data_dir,
        transform=test_transforms
    )

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return test_loader