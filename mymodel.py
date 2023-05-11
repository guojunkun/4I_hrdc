import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import os


# 建立自己的dataset
class HRDCDataset(Dataset):
    def __init__(self, csv_path, file_path, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.device = torch.device("cpu")
        # csv_path = "C:\Users\androidcat\Desktop\cancer_classification\Warwick QU Dataset (Released 2016_07_08)\Grade_train.csv"
        self.file_path = file_path
        self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  # self.data_info.iloc[1:,0表示读取第一列，从第二行开始一直读取到最后一行
        # 第四列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[1:, 1])

        # 计算 length
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]
        image_path = self.file_path + '/' + single_image_name
        # cv2 read data
        input_image = cv2.imread(image_path, 1)
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(input_image, (512, 512))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1)

        image = image.to(self.device, torch.float)

        # PIL读取图像文件(暂时不用）
        # img_as_img = Image.open(self.file_path + '/' + single_image_name)
        #
        # # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        # # if img_as_img.mode != 'L':
        # #     img_as_img = img_as_img.convert('L')
        #
        # # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        # # print(img_as_img.size)
        # transform = transforms.Compose([
        #     transforms.Resize((224,224)),
        #     transforms.ToTensor()
        # ])
        # image = transform(input_image)
        # print(img_as_img.size())
        # 得到图像的 label
        labelnum = self.label_arr[index]
        if labelnum == '0':
            label = 0
        else:
            label = 1

        return image, label# , image_path # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len


class CreatedDataset(Dataset):
    def __init__(self, file_path, label):
        self.label = label
        self.file_path = file_path
        self.files = os.listdir(file_path)
        self.data_num = len(self.files)
        self.device = torch.device("cpu")

    def __getitem__(self, index):
        single_image_name = self.files[index]
        image_path = self.file_path + '/' + single_image_name
        # cv2 read data
        input_image = cv2.imread(image_path, 1)
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(input_image, (512, 512))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1)

        image = image.to(self.device, torch.float)
        label = self.label
        return image, label

    def __len__(self):
        return self.data_num
