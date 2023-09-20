import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 下载数据集，并简单处理保存到 Data 文件夹中
# 目前的图像处理数据集主要 使用 cifar10/100， mnist， fashionMnist， flowers，  ImageNet数据集太大了，以后在说吧

def load_cifar_10():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([     
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径   
    train_data = datasets.CIFAR10(root="./Data", train=True, transform=transform_train, download=True)
    test_data =datasets.CIFAR10(root="./Data", train=False, transform=transform_test, download=True)



def load_MNIST():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])])

    transform_test = transforms.Compose([     
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])])

    train_data = datasets.MNIST(root="./Data/MNIST", train=True, transform=transform_train, download=True)
    test_data =datasets.MNIST(root="./Data/MNIST", train=False, transform=transform_test, download=True)



def load_FashionMNIST():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])])

    transform_test = transforms.Compose([     
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])])
    
    train_data = datasets.FashionMNIST(root="./Data/FashionMNIST", train=True, transform=transform_train, download=True)
    test_data =datasets.FashionMNIST(root="./Data/FashionMNIST", train=False, transform=transform_test, download=True)


def load_cifar_100():

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_test = transforms.Compose([     
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_data = datasets.CIFAR100(root="./Data", train=True, transform=transform_train, download=True)
    test_data =datasets.CIFAR100(root="./Data", train=False, transform=transform_test, download=True)




if __name__ == "__main__":
    # load_cifar_10()
    # load_MNIST()
    # load_FashionMNIST()
    load_cifar_100()
    
