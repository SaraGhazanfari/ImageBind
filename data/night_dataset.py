import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from torch.utils.data import DataLoader


def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
    if preprocess == "DEFAULT":
        t = transforms.Compose([
            transforms.Resize((load_size, load_size), interpolation=interpolation),
            transforms.ToTensor()
        ])
    elif preprocess == "DISTS":
        t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    elif preprocess == "SSIM" or preprocess == "PSNR":
        t = transforms.ToTensor()
    else:
        raise ValueError("Unknown preprocessing method")
    return lambda pil_img: t(pil_img.convert("RGB"))


class NightDataset(Dataset):

    def __init__(self, root_dir, batch_size, num_workers=4, split: str = "test_imagenet",
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: str = "DEFAULT", **kwargs):
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6]  # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.batch_size = batch_size
        self.interpolation = interpolation
        self.n_train_files = 15941
        self.preprocess_fn = get_preprocess_fn(preprocess, 224, self.interpolation)
        self.means = (0.0000, 0.0000, 0.0000)
        self.stds = (1.0000, 1.0000, 1.0000)

        if self.split == "train" or self.split == "val":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        elif split == 'test':
            self.csv = self.csv[self.csv['split'] == split]
        else:
            raise ValueError(f'Invalid split: {split}')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
        img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
        img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
        return img_ref, img_left, img_right, p, id

    def get_dataloader(self):
        shuffle = True if self.split == 'train' else False
        return DataLoader(self, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle), len(self.csv)
