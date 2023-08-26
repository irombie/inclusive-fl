from argparse import Namespace
import os
import random
from typing import Tuple, Union, Dict, List
import os, wget, zipfile

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import tarfile, sys
from parse import parse

# UKTFace

class UTKFaceDataset(Dataset):
    def __init__(self, directory, zfile, extract_dir, transform, label_type="ethnicity"):
        """
        Returns utkface dataset downloaded from link https://susanqq.github.io/UTKFace/.
        Download the aligned and cropped dataset (107 MB) and add it to the data folder
        with name utkface.tar.gz.
        Other helper references: https://github.com/AryaHassanli/

        :params directory: directory where the images are located
        :params zfile: relative path from home folder to the zip file stored under data
        :params extract_dir: main directory where the UTKFace folder will be stored
        :params transform: image transformation for UTKFace

        :returns: dataset that can be used for training UTKFace
        """
        self.directory = directory
        self.transform = transform
        self.label_type = label_type
        self.labels = []
        self.images = []

        if os.path.isdir(directory) and len(os.listdir(directory)) > 0:
            print('UTK Already Exists on', self.directory, ' / We will use it!')
        else:
            print('Could not find UTK on', directory)
            print('Looking for ', zfile)
            if os.path.exists(zfile):
                print(zfile, 'is found. Trying to extract:')
                try:
                    tar = tarfile.open(zfile, "r:gz")
                    tar.extractall(path=extract_dir)
                    tar.close()
                    print('Successfully extracted')
                except tarfile.TarError:
                    sys.exit('Extract Failed!')
            else:
                sys.exit('UTK Zip file not found!')

        for i, file in enumerate(os.listdir(extract_dir+'/UTKFace')):
            file_labels = parse('{age}_{gender}_{ethnicity}_{}.jpg', file)
            if file_labels is not None:
                if int(file_labels['age']) > 120 or int(file_labels['gender']) > 1:
                    continue

                image = Image.open(os.path.join(extract_dir+'/UTKFace', file))
                image = self.transform(image)

                self.images.append(image)
                self.labels.append({
                    'age': self.convert_age_to_range(int(file_labels['age'])),
                    'gender': int(file_labels['gender']),
                    'ethnicity': int(file_labels['ethnicity']),
                })
    
    def convert_age_to_range(self, age):
        label = 0
        if age <= 10:
            label = 0
        elif 11 <= age <= 20:
            label = 1
        elif 21 <= age <= 30:
            label = 2
        elif 31 <= age <= 40:
            label = 3
        elif 41 <= age <= 60:
            label = 4
        elif 61 <= age <= 80:
            label = 5
        elif age >= 81:
            label = 6
        return label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]

        try: # accepts age/gender/ethnicity labels
            labels = self.labels[idx][self.label_type]
        except:
            print("Wrong Label Type provided")
            return

        return image, labels

def get_utkface(data_dir, zfile, extract_dir, apply_transform, label_type="ethnicity"):
    """
        Returns train/test/validation utkface datasets.

        :params data_dir: directory where the images are located
        :params zfile: relative path from home folder to the zip file stored under data
        :params extract_dir: main directory where the UTKFace folder will be stored
        :params apply_transform: image transformation for UTKFace

        :returns: train/test/validation utkface datasets that can be used for training UTKFace
        """
    dataset = UTKFaceDataset(
        directory=data_dir, 
        zfile=zfile, 
        extract_dir=extract_dir, 
        transform=apply_transform,
        label_type=label_type,
    )

    train_len = int(len(dataset) * 0.8)
    validate_len = int(len(dataset) * 0.1)
    test_len = int(len(dataset) - train_len - validate_len)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_len, validate_len, test_len,]
    )

    return train_dataset, test_dataset, valid_dataset

# CelebA

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    # Reference: https://datahacker.rs/015-pytorch-building-a-smile-detector-with-pytorch/
    # Reference: https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb
    def __init__(self, df, img_dir, transform, label_type):
        """
            df: contains image jpg number and label values for 40 attributes,
                split into train/test/val apriori
            img_dir: directory where actual jpegs are stored
            transform: image transform, cannot be none
            label_type: currently supports gender and smiling labels
        """

        assert transform != None, "Transform cannot be none"

        self.img_names = df.index.values

        if label_type == "gender":
            self.y = df['Male'].values
        elif label_type == "smiling":
            self.y = df['Smiling'].values
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        img = self.transform(img)
        label = self.y[index]

        if label == -1:
            label = np.int64(0.0)
        else:
            label = np.int64(1.0)

        return img, label

    def __len__(self):
        return self.y.shape[0]

def get_celeba(data_dir, label_type, apply_transform):
    """
        data_dir: stores core files for CelebA
        label_type: either gender or smiling
        apply_transform: image transform, cannot be none
    """
    base_url = "https://graal.ift.ulaval.ca/public/celeba/"

    file_list = [
        "img_align_celeba.zip",
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_align_celeba.txt",
        "list_eval_partition.txt",
    ]

    os.makedirs(data_dir, exist_ok=True)

    for file in file_list:
        url = f"{base_url}/{file}"
        if not os.path.exists(f"{data_dir}/{file}"):
            wget.download(url, f"{data_dir}/{file}")

    with zipfile.ZipFile(f"{data_dir}/img_align_celeba.zip", "r") as ziphandler:
        ziphandler.extractall(data_dir)

    attr_path = f"{data_dir}/list_attr_celeba.txt"
    df = pd.read_csv(attr_path, sep="\s+", skiprows=1,)

    train, valid, test = np.split(
        df.sample(frac=1), 
        [int(.6*len(df)), int(.8*len(df))],
    )

    img_dir = f"{data_dir}/img_align_celeba"

    train_dataset = CelebaDataset(train, img_dir, apply_transform, label_type)
    test_dataset = CelebaDataset(test, img_dir, apply_transform, label_type)
    valid_dataset = CelebaDataset(valid, img_dir, apply_transform, label_type)

    return train_dataset, test_dataset, valid_dataset
