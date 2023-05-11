from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
from matplotlib import pyplot as plt
from mymodel import HRDCDataset
from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image


def brightnessEnhancement(img_name):#亮度增强
    image = Image.open(img_name)
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    # brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(img_name):  # 对比度增强
    image = Image.open(img_name)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    # contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def rotation(img_name):
    img = Image.open(img_name)
    random_angle = np.random.randint(-2, 2)*90
    if random_angle==0:
     rotation_img = img.rotate(-90) #旋转角度
    else:
        rotation_img = img.rotate( random_angle)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def flip(img_name):   #翻转图像
    img = Image.open(img_name)
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def createImage(imageDir, saveDir, i):
    saveName = saveDir + '/' + "cesunnew" + str(i) + ".jpg"
    saveImage = contrastEnhancement(imageDir)
    saveImage.save(saveName)
    saveName1 = "flip" + str(i) + ".jpg"
    saveImage1 = flip(imageDir)
    saveImage1.save(os.path.join(saveDir, saveName1))
    saveName2 = "brightnessEnew" + str(i) + ".jpg"
    saveImage2 = brightnessEnhancement(imageDir)
    saveImage2.save(os.path.join(saveDir, saveName2))
    saveName3 = "rotatenew" + str(i) + ".jpg"
    saveImage = rotation(imageDir)
    saveImage.save(os.path.join(saveDir, saveName3))


image_dir = "1-Images/1-Training Set"
csv_dir = "2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv"
train_set = HRDCDataset(csv_dir,image_dir)
# train_set = datasets.ImageFolder(train_dir)
train_iter = DataLoader(train_set, batch_size=1, shuffle=True)
i=0
path0 = "created_data/0"
path1 = "created_data/1"
for X, y, path in train_iter:
    i=i+1
    if y[0] == 0:
        createImage(path[0], path0, i)
    else:
        createImage(path[0], path1, i)
