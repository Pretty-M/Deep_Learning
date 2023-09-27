import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from vgg import vgg

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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

# ********************* 训练模型 **************************** #
def train_model(model, train_loader, test_loader, device):
    # 训练参数
    num_epochs = 50
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cwd = os.getcwd()
    # save_path = cwd + '/Result/vggnet/vggmodel_cifar100.pkl'    # 保存模型和模型名字
    save_path = cwd + '/Result/vggnet/vggmodel_flower.pkl'    # 保存模型和模型名字

    # 训练模型 
    print('Start training...')
    
    train_acc_list = []
    train_loss_list = []
    val_acc_lost = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)

        for images, labels in train_bar:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels).sum().item()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, loss)
        
        train_acc = train_acc / len(train_loader.dataset)
        running_loss = running_loss / len(train_loader.dataset)

        train_acc_list.append(train_acc)
        train_loss_list.append(running_loss)

        print('[Epoch %d] Train Loss: %.4f, Train Acc: %.3f%%' % (epoch + 1, running_loss, 100*train_acc))

        # validate
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        val_accurate = acc / len(val_loader.dataset)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
            print('模型保存成功！')






if __name__ == "__main__":

    path0 = os.getcwd()
    save_path = path0 + "/Result/alexnet"


    # **************** 使用cifar100数据训练模型 ******************* #
    # # 加载数据集
    # train_loader, val_loader = load_CIFAR100(batch_size=128, resize=227)

    # # 实例化模型
    # model_name = "vgg16"
    # net = vgg(model_name=model_name, num_classes=100, init_weights=True)
    # net.to(device)


    # **************** 使用flower数据训练模型 ******************* #
    # 加载数据集
    train_loader, val_loader = load_Flowers(batch_size=128, resize=227)
    # 实例化模型
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)

    # 训练模型
    train_model(net, train_loader, val_loader, device)


   


    # 可视化结果（画误差曲线， 输出判决结果）

    

