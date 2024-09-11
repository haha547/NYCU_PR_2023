import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import pandas as pd


class Synth90kDataset(Dataset):
    CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, paths=None, mode = None, img_height=32, img_width=100, transform = None):
        if root_dir and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and paths:
            texts = None

        self.mode = mode
        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def _load_from_raw_files(self, root_dir, mode):
        if mode != "test":
            df = pd.read_csv(os.path.join(root_dir, "annotations.csv"))
        else:
            df = pd.read_csv("sample_submission.csv")
        paths = df[["filename"]].to_numpy()
        texts = None
        if mode == "train":
            texts = df[["label"]].to_numpy()

        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index][0]
        try:
            temp = os.path.join(self.mode, path)
            image = Image.open(temp).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            text = self.texts[index][0]
            print(text)
            return self[index + 1]


        if self.transform is not None:
            image = self.transform(image)

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts is not None:
            text = self.texts[index][0]
            
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image



def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
