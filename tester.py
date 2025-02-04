#!/usr/bin/env python3
import os
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import numpy as np
from tqdm import tqdm

# Import your custom modules
from resnet_unet.resnet_unet_model import ResNetUNet
from testDataset import TestDataset


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test Processing: Run inference on test dataset and save predictions."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/",
        help="Directory containing the input images."
    )
    parser.add_argument(
        "--csv_data_path",
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest2/dataset_0/test.csv",
        help="CSV file path listing the test image names."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="outputs/0",
        help="Directory where the output .npy files will be saved."
    )
    parser.add_argument(
        "--chkpt_path",
        type=str,
        default="./ckpts/model.pth",
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing the test dataset."
    )
    return parser.parse_args()


class TestProcessing:
    def __init__(self, args):
        self.args = args
        self.img_transform = Compose([ToTensor()])
        self.model = self._init_model(args.chkpt_path)
        self.test_dataset = TestDataset(args.data_dir, args.csv_data_path, self.img_transform)

    def _init_model(self, chkpt_path: str) -> ResNetUNet:
        """
        Initialize the model, load the checkpoint, and set it to evaluation mode.
        """
        model = ResNetUNet(n_class=3).cuda()
        model.load_state_dict(torch.load(chkpt_path))
        model.eval()
        return model

    def save_imgs(self, model_output: torch.Tensor, data_paths, epsilon: float = 1e-9):
        """
        Normalize the model output per sample and save each result as a .npy file.

        Args:
            model_output (torch.Tensor): Tensor of shape (N, C, H, W).
            data_paths (list): List of image path strings corresponding to the batch.
            epsilon (float): Small constant to avoid division by zero.
        """
        # Rearrange tensor to shape (N, H, W, C) and convert to a NumPy array.
        output_np = model_output.permute(0, 2, 3, 1).cpu().numpy()
        # Compute the L2 norm along the channel dimension.
        norms = np.linalg.norm(output_np, axis=-1, keepdims=True)
        # Normalize the vectors (adding epsilon to avoid division by zero).
        normal_vectors = output_np / (norms + epsilon)

        for i, data_path in enumerate(data_paths):
            # Replace "crop.jpg" with "predict.npy" in the file name.
            dest_path = os.path.join(self.args.dest_dir, str(data_path).replace("crop.jpg", "predict.npy"))
            # Ensure the destination directory exists.
            Path(Path(dest_path).parent).mkdir(parents=True, exist_ok=True)
            np.save(dest_path, normal_vectors[i])

    def process(self):
        """
        Run the model on the test dataset and save the predictions.
        """
        test_dl = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False
        )
        with torch.no_grad():
            for data in tqdm(test_dl, desc="Processing"):
                # Move images to GPU and extract image paths.
                img = data["image"].cuda()
                img_paths = data["name"]
                # Get model output (assuming the model returns a tuple and we need the first element).
                output = self.model(img)[0]
                self.save_imgs(output, img_paths)


def main():
    args = parse_args()
    # Create the destination directory if it does not exist.
    Path(args.dest_dir).mkdir(exist_ok=True, parents=True)
    processor = TestProcessing(args)
    processor.process()


if __name__ == "__main__":
    main()