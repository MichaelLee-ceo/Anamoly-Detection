import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from dataset import load_data
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist", type=str)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--ckpt_pth', default="./checkpoint/", type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("> Using:", device)

dataset_name = args.dataset
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
img_size = args.img_size
latent_dim = args.latent_dim
ckpt_pth = args.ckpt_pth

os.makedirs(args.ckpt_pth, exist_ok=True)

# train_data = ImageFolder(root=train_path, transform=data_transform)
# test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

''' Load trainin data '''
train_dataloader, test_dataloader = load_data(dataset_name=dataset_name, normal_class=0, args=args)

''' Autoencoder model '''
model = AutoEncoder_CNN(dim=latent_dim, large=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    loss_list = []

    for idx, (img, label) in enumerate(train_dataloader):
        img = img.to(device)
        output = model(img)
        
        loss = loss_function(img, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    print('{} | Epoch [{}/{}], loss:{:.4f}'.format(dataset_name, epoch + 1, num_epochs, np.mean(loss_list)))
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), args.ckpt_pth + dataset_name + ".pth")