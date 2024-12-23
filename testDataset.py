import torch.nn as nn
import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader



class TestDataset(nn.Module):
    def __init__(self, root_data, csv_file, transform):
        self.root_data = root_data
        self.transform = transform
        self.csv_file = csv_file
        self.data_info = pd.read_csv(self.csv_file, header=None)


    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_data, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        return {"image": image, "name": self.data_info.iloc[idx, 0]}

    


if __name__ == '__main__':
    root_data = '/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormal'
    csv_file = '/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest/dataset_0/test.csv'

    transform = Compose([
        ToTensor()
    ])

    test_dataset = TestDataset(root_data, csv_file, transform)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    for idx, data in enumerate(test_dl):
        print(data["image"].shape, data["name"][0])
        break
