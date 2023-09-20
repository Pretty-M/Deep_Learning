import torch
import torch.nn as nn


# ***************************** AlexNet 模型（这个尺寸是为Imagenet数据集服务的） ****************************** #
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000,  init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积层： 输入通道数为3，输出通道数为96，卷积核大小为11，步长为4
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),        # shape: (batch, 96, 55, 55)
            nn.ReLU(inplace=True),
            # 第一个池化层： 输入通道数为96，输出通道数为96，池化核大小为3，步长为2
            nn.MaxPool2d(kernel_size=3, stride=2),                                      # shape: (batch, 96, 27, 27)

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),      # shape: (batch, 256, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                      # shape: (batch, 256, 13, 13)

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),     # shape: (batch, 384, 13, 13)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),     # shape: (batch, 384, 13, 13)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),     # shape: (batch, 256, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      # shape: (batch, 256, 6, 6)
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        x = self.features(x)        # shape: (batch, 256, 6, 6)
        x = x.view(x.size(0), 256 * 6 * 6)      # shape: (batch, 256 * 6 * 6)
        x = self.fc(x)               # shape: (batch, 10)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 注意： 全尺寸的 AlexNet 模型训练数据集时，需要将图片resize到  227*227



# ****************************** 小尺寸的可以适用于 CIFAR10/FashionMnist 数据集合
# class AlexNetmini(nn.Module):
#     def __init__(self, num_classes=10,  init_weights=False):
#         super(AlexNet, self).__init__()
