#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, vgg11_bn


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
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
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class SmallCNN(nn.Module):
    def __init__(self, args, num_classes):
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
    def __init__(self, num_classes: int, args) -> None:
        super().__init__()
        self.vgg = vgg11_bn(pretrained=False)
        self.classifier = nn.Linear(1000, num_classes)
        self.dataset = args.dataset

    def forward(self, x):
        x = self.vgg(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, args) -> None:
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.classifier = nn.Linear(1000, num_classes)
        self.dataset = args.dataset

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv_res1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class ResNet9(nn.Module):
    def __init__(self, num_classes: int, args):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dataset = args.dataset

        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)

        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
