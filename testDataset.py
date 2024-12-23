import torch.nn as nn
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor


class TestDataset(nn.Module):
    def __init__(self, root_data, csv_file, transform):
        self.root_data = root_data
        self.transform = transform
        self.csv_file = csv_file
        data_info = pd.read_csv(self.csv_file, header=None)


    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_data, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        return {"image": image, "name": self.data_info.iloc[idx, 0]}
    


if __name__ == '__main__':
    root_data = 'data/cropped'
    csv_file = 'data/test.csv'

    transform = Compose([
        ToTensor()
    ])

    test_dataset = TestDataset(root_data, csv_file, transform)
    print(test_dataset[0])
    print(len(test_dataset))