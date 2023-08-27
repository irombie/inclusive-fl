from argparse import Namespace
import os
import random
from typing import Tuple, Union, Dict, List
import os, wget, zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from dataloader_utils import get_celeba, get_utkface
import tarfile, sys
from PIL import Image
from parse import parse

import unittest 


class TestCelebaDataLoader(unittest.TestCase):
    def test_loading(self):
        data_dir = 'data/celeba'

        mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
        std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

        apply_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        label_type = "gender"
        train_dataset, test_dataset, valid_dataset = get_celeba(
            data_dir, 
            label_type, 
            apply_transform
        )

        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        features, labels = next(iter(trainloader))

        assert features.shape == torch.Size([64, 3, 224, 224])


class TestUTKFaceDataLoader(unittest.TestCase):
    def test_loading(self):
        data_dir = 'data/UTKFace'
        apply_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.49,), (0.23,))
        ])

        train_dataset, test_dataset, valid_dataset = get_utkface(
            data_dir=data_dir, 
            zfile='data/utkface.tar.gz', 
            extract_dir='data', 
            apply_transform=apply_transform,
            label_type="ethnicity"
        )

        print('Dataset Lengths', len(train_dataset), '/', len(test_dataset), '/', len(test_dataset))

        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        features, labels = next(iter(trainloader))

        assert features.shape == torch.Size([64, 3, 128, 128])

if __name__ == '__main__':
    unittest.main()