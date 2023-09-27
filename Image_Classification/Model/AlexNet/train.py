import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from alexnet import AlexNet


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# ************************** 处理数据集----CIFAR10 ********************
def load_CIFAR10(batch_size, resize=227):

    transForms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if resize:
        transForms.insert(0, transforms.Resize(resize))
    transForms = transforms.Compose(transForms)

    cwd = os.getcwd()
    train_set = datasets.CIFAR10(root=cwd+'/Data', train=True, download=True, transform=transForms)
    val_set = datasets.CIFAR10(root=cwd+'/Data', train=False, download=True, transform=transForms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def get_labels(labels):
    """标签转换"""
    text_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # 类别名称
    return [text_labels[int(i)] for i in labels]


# ************************** 处理数据集----flower ********************
def load_Flowers(batch_size, resize=224):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((resize, resize)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    path = os.getcwd()
    image_path = path + "/Data/flower_data"

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)


    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    return train_loader, val_loader


# ************************** 处理数据集----CIFAR100 ********************
def load_CIFAR100(batch_size, resize=227):
    data_transform = {
        # 进行数据增强的处理
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
 
        "val": transforms.Compose([transforms.Resize((resize, resize)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    }
    cwd = os.getcwd()
    train_set = datasets.CIFAR100(root=cwd+'/Data', train=True, download=True, transform=data_transform["train"])
 
    val_set = datasets.CIFAR100(root=cwd+'/Data', train=False, download=False, transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
 
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader





# ************************ 验证模型 *******************************
def evaluate_accuracy(data_iter, net, device=None):

    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定 device 就使用 net 的 device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    for step, data in enumerate(data_iter):
        X, y = data
        X, y = X.to(device), y.to(device)
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
        n += y.shape[0]
    return acc_sum / n


# ************************* 训练模型 ******************************
def train_model(model, train_loader, test_loader):
  

    # 训练参数
    num_epochs = 50
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型 
    print('Start training...')
    
    for epoch in range(num_epochs):

        train_loss, train_acc, n = 0.0, 0.0, 0
        start = time.time()     # 记录训练时间

        for step, data in enumerate(train_loader, start=0):
            images, labels = data       # data 是个列表 [数据, 标签]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            train_loss += l.cpu().item()
            train_acc += (outputs.argmax(dim=1) == labels).sum().cpu().item()
            n += labels.shape[0]

        # 每个epoch测试一次
        test_acc = evaluate_accuracy(test_loader, model)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss / n, train_acc / n, test_acc, time.time() - start))

    print('Finished training.')

    return model








if __name__ == "__main__":

    path0 = os.getcwd()
    save_path = path0 + "/Result/alexnet"


    # **************** 使用cifar10数据训练模型 ******************* #
    # # 加载数据集
    # train_loader, val_loader = load_CIFAR10(batch_size=128, resize=227)
    # # 实例化模型
    # net = AlexNet(num_classes=10, init_weights=True)
    # net.to(device)

    # **************** 使用cifar100数据训练模型 ******************* #
    # 加载数据集
    train_loader, val_loader = load_CIFAR100(batch_size=128, resize=227)
    # 实例化模型
    net = AlexNet(num_classes=100, init_weights=True)
    net.to(device)


    # **************** 使用flower数据训练模型 ******************* #
    # # 加载数据集
    # train_loader, val_loader = load_Flowers(batch_size=128, resize=227)
    # # 实例化模型
    # net = AlexNet(num_classes=5, init_weights=True)
    # net.to(device)

    # 训练模型
    trained_model= train_model(net, train_loader, val_loader)

    # 保存模型
    # torch.save(net.state_dict(), save_path + '/alexnet_cifar10.pkl')
    torch.save(net.state_dict(), save_path + '/alexnet_cifar100.pkl')

    # torch.save(net.state_dict(), save_path + '/alexnet_flower.pkl')
    print('模型保存成功！')

    

    

    # 可视化结果（画误差曲线， 输出判决结果）

    
