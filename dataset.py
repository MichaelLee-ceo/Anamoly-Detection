import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(dataset_name="mnist", normal_class=0, args=None):
    img_size = args.img_size
    batch_size = args.batch_size

    if dataset_name == "mnist":
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        dataset = datasets.MNIST(root="./data", download=True, train=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.target = [normal_class] * dataset.data.shape[0]
        testset = datasets.MNIST(root="./data", download=False, train=False, transform=img_transform)
    elif dataset_name == "fashionmnist":
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        dataset = datasets.FashionMNIST(root="./data", download=True, train=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.target = [normal_class] * dataset.data.shape[0]
        testset = datasets.FashionMNIST(root="./data", download=False, train=False, transform=img_transform)
        
    print("Normal training data: {}".format(dataset.data.shape))
    print("Testing data: {}".format(testset.data.shape))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    return train_loader, test_loader