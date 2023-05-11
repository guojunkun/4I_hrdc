from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader
from torch import nn
import random
from matplotlib import pyplot as plt
from mymodel import HRDCDataset, CreatedDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
import os
from sklearn.metrics import f1_score
# from model import ResNet34

image_dir = "1-Images/1-Training Set"
csv_dir = "2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv"
train_set = HRDCDataset(csv_dir,image_dir)
t_iter = DataLoader(train_set, batch_size=1)
created0 = CreatedDataset("created_data/0", 0)
created1 = CreatedDataset("created_data/1", 1)
trainval = ConcatDataset([train_set, created1, created0])
train, val = train_test_split(trainval, test_size=0.2, random_state=42)
# train_set = datasets.ImageFolder(train_dir)
train_iter = DataLoader(train, batch_size=64, shuffle=True)
val_iter = DataLoader(val, batch_size=64)
#print(train_set.data_info)
# model1  resnet512_1.pt
# resnet = models.resnet50(pretrained=True)
# resnet.add_module("final_fc", nn.Linear(1000,2))
# model2  new2.pt
# resnet = models.resnet34(pretrained=True)
# resnet.add_module("final_fc", nn.Linear(1000,2))
# model3
resnet = models.resnet34(pretrained=True)
resnet.add_module("final_fc1", nn.Linear(1000, 50))
resnet.add_module("final_fc2", nn.Linear(50, 2))
# print(resnet)
# for name, param in resnet.named_parameters():
#     if name not in ['fc.weight', 'fc.bias', 'final_fc.weight', 'final_fc.bias', 'final_fc1.weight',
#     'final_fc1.bias', 'final_fc2.weight', 'final_fc2.bias']:
#         param.requires_grad = False
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
loss_fun = torch.nn.CrossEntropyLoss()

path = 'model3_2.pt'
if os.path.exists(path):
    loaded_paras = torch.load(path, map_location=torch.device('cpu'))
    resnet.load_state_dict(loaded_paras)
for epoch in range(15):
    resnet.train()  # 训练模式
    # for name, param in resnet.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias', 'final_fc.weight', 'final_fc.bias', 'final_fc1.weight',
    #     'final_fc1.bias', 'final_fc2.weight', 'final_fc2.bias']:
    #         param.requires_grad = False
    train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
    for X, y in train_iter:
        # print(X)
        # print(y)
        # print(type(y[0]))
        optimizer.zero_grad()  # 梯度清零
        output = resnet(X)
        loss = loss_fun(output, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_acc_sum += torch.eq(output.argmax(dim=1), y).sum().item()
        n += y.shape[0]
        batch_count += 1
        if batch_count % 5 == 0:
            print(train_acc_sum / n)

    print('epoch %d, loss %.4f, train acc %.3f' % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n))
    torch.save(resnet.state_dict(), 'model3_3.pt')

    resnet.eval()
    val_loss_sum, val_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
    for X, y in val_iter:
        output = resnet(X)
        loss = loss_fun(output, y)

        val_loss_sum += loss.item()
        val_acc_sum += torch.eq(output.argmax(dim=1), y).sum().item()
        n += y.shape[0]
        batch_count += 1
        if batch_count % 100 == 0:
            print(loss)
            print(y.shape)

    print(' loss %.4f, train acc %.3f' % (val_loss_sum / batch_count, val_acc_sum / n))
torch.save(resnet.state_dict(), 'model3_3.pt')

val_loss_sum, val_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
resnet.eval()
y_true = []
y_pred = []
for X, y in t_iter:
    output = resnet(X)
    loss = loss_fun(output, y)

    val_loss_sum += loss.item()
    val_acc_sum += torch.eq(output.argmax(dim=1), y).sum().item()
    n += y.shape[0]
    batch_count += 1
    y_true.append(y)
    out = output.argmax(dim=1)
    out = out.detach().cpu()
    out = int(out)
    y_pred.append(out)
    if batch_count % 100 == 0:
        print(loss)
        print(y.shape)

print(' loss %.4f, train acc %.3f' % (val_loss_sum / batch_count, val_acc_sum / n))
f1 = f1_score(y_true, y_pred, average='binary')
print(f1)
#
# def denorm(img):
#     for i in range(img.shape[0]):
#         img[i] = img[i]
#     return img
# plt.figure(figsize=(8, 8))
# for i in range(9):
#     img, label = train_set[random.randint(0, len(train_set))]
#     img = denorm(img)
#     img = img.permute(1, 2, 0)
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()