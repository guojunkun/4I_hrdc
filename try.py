from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
from matplotlib import pyplot as plt
from mymodel import HRDCDataset, CreatedDataset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import os
# from model import ResNet34

image_dir = "1-Images/1-Training Set"
csv_dir = "2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv"
train_set = HRDCDataset(csv_dir,image_dir)
created0 = CreatedDataset("created_data/0", 0)
created1 = CreatedDataset("created_data/1", 1)
train = ConcatDataset([train_set, created1, created0])
train_iter = DataLoader(train, batch_size=64, shuffle=True)
num = 0
for X, y in train_iter:
    num += y.shape[0]
print(num)
