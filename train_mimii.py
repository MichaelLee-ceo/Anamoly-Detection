import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torchvision.utils import make_grid, save_image
from dataset import load_data
from models.model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mimii", type=str)
parser.add_argument('--abnormal_class', default="", type=str)
parser.add_argument('--num_db', default="6", type=str)
parser.add_argument('--machine_id', default="00", type=str)
parser.add_argument('--machine_type', default="fan", type=str)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--ckpt_pth', default="./checkpoint/", type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

# Seed everything
seed_everything(seed=args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("> Using:", device)

dataset_name = args.dataset
num_epochs = args.num_epochs
lr = args.lr
latent_dim = args.latent_dim
ckpt_pth = args.ckpt_pth

os.makedirs(ckpt_pth, exist_ok=True)

# loop for all datasets
machine_types = ["fan"]
# num_dbs = ["6", "0", "min6"]
# machine_ids = ["00", "02", "04", "06"]
num_dbs = ["0"]
machine_ids = ["00"]

for num_db in num_dbs:
    for machine_id in machine_ids:
        args.machine_id = machine_id
        args.num_db = num_db
        machine_info = args.machine_type + "_id_" + args.machine_id + "_db_" + args.num_db
        ckpt_pth = args.ckpt_pth + dataset_name + "_" + machine_info

        print("\n========== Training on: {} ==========".format(machine_info))

        ''' Load trainin data '''
        train_dataloader, test_dataloader = load_data(dataset_name=dataset_name, normal_class=0, args=args)
        save_image(make_grid(next(iter(train_dataloader))[:32][0]), "./samples/" + dataset_name + "_" + machine_info + "_normal.png")

        ''' Autoencoder model '''
        model = AutoEncoder_CNN(channel=3, dim=latent_dim, large=True).to(device)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)

        for epoch in range(num_epochs):
            model.train()
            loss_list = []

            for idx, (img, label) in enumerate(tqdm(train_dataloader, leave=False)):
                img = img.to(device)
                output = model(img)
                
                loss = loss_function(img, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            tqdm.write('{} | Epoch [{}/{}], loss:{:.4f}'.format(dataset_name, epoch + 1, num_epochs, np.mean(loss_list)))
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                torch.save(model.state_dict(), ckpt_pth + ".pth")