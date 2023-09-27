import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm
from resnet import resnet34

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def load_Flowers(batch_size, resize=224):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize((resize, resize)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    path = os.getcwd()
    image_path = path + "/Data/flower_data"

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
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



# ********************* 训练模型 **************************** #
def train_model(model, train_loader, test_loader, device):

    # 训练参数
    num_epochs = 50
    loss_function = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    cwd = os.getcwd()
    save_path = cwd + '/Result/resnet/resnetmodel_cifar100.pkl'    # 保存模型和模型名字
    # save_path = cwd + '/Result/resnet/resnetmodel_flower.pkl'    # 保存模型和模型名字

    print('Start training...')

    best_acc = 0.0
    train_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        net.train()

        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss)
        
        running_loss /= len(train_loader)
        train_loss_list.append(running_loss)            # 记录每个epoch的loss 加入列表

        # validate the model
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, num_epochs)

        val_accurate = acc / len(test_loader.dataset)
        val_acc_list.append(val_accurate)               # 记录每个epoch的acc 加入列表

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss, val_accurate))
        

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
            print('模型保存成功！')
        
        








if __name__ == "__main__":


     # **************** 使用flower数据训练模型 ******************* #
    # 加载数据集
    # train_loader, val_loader = load_Flowers(batch_size=128, resize=227)
    # # 实例化模型
    # net = resnet34(num_classes=5)
    # net.to(device)

    # **************** 使用cifar100数据训练模型 ******************* #
    # 加载数据集
    train_loader, val_loader = load_CIFAR100(batch_size=128, resize=227)
    # 实例化模型
    net = resnet34(num_classes=100)
    net.to(device)



    # 训练模型
    train_model(net, train_loader, val_loader, device)

