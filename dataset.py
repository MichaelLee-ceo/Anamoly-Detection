import os
import random
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

def get_data_transforms(size, crop_size):
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms

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
        print("Normal training data: {}".format(dataset.data.shape))
        print("Testing data: {}".format(testset.data.shape))

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

    elif dataset_name == "windmill":
        img_transform = transforms.Compose([
            transforms.CenterCrop(720),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = WindDataset(root="./data/Fan/", train=True, transform=img_transform, args=args)
        testset = WindDataset(root="./data/Fan/", train=False, transform=img_transform, args=args)
        
        print("Normal training data: {}, {}".format(len(dataset.data), dataset.data[0].size))
        print("Testing data: {}, {}".format(len(testset.data), testset.data[0].size))
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)

    return train_loader, test_loader

def getImage(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = Image.open(os.path.join(path, file)).convert('RGB')
            files.append(img)
    return files

class WindDataset(Dataset):
    def __init__(self, root, train=True, transform=None, args=None):
        self.root = root
        self.transform = transform
        self.normal = getImage(os.path.join(self.root, "normal"))
        self.abnormal = getImage(os.path.join(self.root, "abnormal"))

        random.shuffle(self.normal)
        num_abnormal = len(self.abnormal)

        if train:
            self.data = self.normal[num_abnormal:]
            self.label = [0] * len(self.data)
        else:
            self.data = self.normal[:num_abnormal] + self.abnormal
            self.label = [0] * len(self.normal[:num_abnormal]) + [1] * len(self.abnormal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, self.label[index]