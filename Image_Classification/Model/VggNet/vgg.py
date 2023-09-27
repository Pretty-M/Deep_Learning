import torch
import torch.nn as nn

# ********************* VGG 模型 ********************** #

class VGGmodel(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGGmodel, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),       # 全连接层
            nn.ReLU(True),
            nn.Dropout(p=0.5),       # 减小过拟合

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N*3*224*224
        x = self.features(x)        # shape: (batch, 512, 7, 7)
        x = torch.flatten(x, start_dim=1)       # shape: (batch, 512 * 7 * 7)
        x = self.classifier(x)         # shape: (batch, 1000)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):        # 若是卷积层，则利用xavier进行初始化
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):      # 全连接层
                nn.init.xavier_normal(m.weight)      # 权重初始化
                nn.init.constant_(m.bias, 0)        # 偏置初始化

# 以列表形式记录vgg各个模型的参数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_feature(cfg: list):       # 提取特征函数
    layers = []     # 用于存放创建每一层结构
    in_channels = 3     # RGB
    for v in cfg:
        if v == 'M':    # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]     # 池化层
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)     # 卷积层
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v     # 卷积后，卷积层输入channel变成上一层的channel
    return nn.Sequential(*layers)


def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print('Warning: model number {} not in cfgs dict!'.format(model_name))
        exit(-1)

    model = VGGmodel(make_feature(cfg), **kwargs)
    return model