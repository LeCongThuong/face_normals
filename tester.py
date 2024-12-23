import torch
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor
from resnet_unet.resnet_unet_model import ResNetUNet
import cv2
import numpy as np
from PIL import Image
from testDataset import TestDataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import os
from pathlib import Path
from tqdm import tqdm

@dataclass
class Config:
    data_dir: str = '/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/'
    csv_data_path: str = "/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest2/dataset_1/test.csv"
    dest_dir: str = "outputs_4"
    chkpt_path: str = "data/model.pth"
    batch_size: int = 32


class TestProcessing:
  def __init__(self, cfg):
    self.cfg = cfg
    img_transform = Compose([
                          ToTensor()
                        ])
    self.model = self._init_model(cfg.chkpt_path)
    self.test_dataset = TestDataset(cfg.data_dir, cfg.csv_data_path, img_transform)
    
    
  def save_imgs(self, model_output, data_paths, epsilon = 1e-9):
    model_output_perm = model_output.permute(0, 2, 3, 1)
    model_output_np = model_output_perm.cpu().numpy() 
    norms = np.linalg.norm(model_output_np, axis=-1, keepdims=True)  # shape: (BS, H, W, 1)
    # Add a small epsilon to avoid division by zero
    normal_vectors_np = model_output_np / (norms + epsilon)

    num_img = model_output.shape[0]
    for i in range(num_img):
      data_path = data_paths[i]
      normal_v = normal_vectors_np[i]
      dest_path = os.path.join(cfg.dest_dir, str(data_path).replace("crop.jpg", "predict.npy"))
      Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
      np.save(dest_path, normal_v)
 
    
  def _init_model(self, chkpt_path):
    model = ResNetUNet(n_class = 3).cuda()
    model.load_state_dict(torch.load(chkpt_path))
    model.eval()
    return model 

  def process(self):
    test_dl = DataLoader(self.test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
      for idx, data in tqdm(enumerate(test_dl)):
        img, img_paths = data["image"].cuda(), data["name"]
        out = self.model(img)[0]
        self.save_imgs(out, img_paths)

if __name__ == "__main__":
  cfg  = Config()
  Path(cfg.dest_dir).mkdir(exist_ok=True, parents=True)
  test_process = TestProcessing(cfg)
  test_process.process()
  




