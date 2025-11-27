#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, vgg11_bn


class LogisticRegression(nn.Module):
    """
    Logistic regression model for the synthetic dataset.
    """

    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    """
    This is an hardcoded MLP model for the synthetic dataset.
    The hyperparameters are not optimized.
    """

    def __init__(self, num_classes, num_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNFashionMNIST(nn.Module):
    def __init__(self, num_classes: int):
        super(CNNFashionMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()

        self.normalization = nn.BatchNorm2d

        self.activation = nn.ReLU()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            self.normalization(16),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            self.normalization(32),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            self.normalization(64),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.normalization(128),
            self.activation,
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.normalization(256),
            self.activation,
            nn.MaxPool2d(2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 if num_classes == 10 else 1024, 256),
            nn.Dropout(),
            self.activation,
            nn.Linear(256, 256),
            nn.Dropout(),
            self.activation,
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class VGG(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.vgg = vgg11_bn(weights=None)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.vgg(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.classifier = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class ResNet9(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, self.num_classes, bias=False),
            Mul(0.2),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
