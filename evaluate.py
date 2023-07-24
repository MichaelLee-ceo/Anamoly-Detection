import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from dataset import load_data
from models.model import *
from utils import *
from torchvision.utils import make_grid, save_image
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="windmill", type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--ckpt_pth', default="./checkpoint/", type=str)
parser.add_argument('--figure_dir', default="./figures/", type=str)
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

# Seed everything
seed_everything(seed=args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("> Using:", device)

invTrans = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
])

dataset_name = args.dataset_name
batch_size = args.batch_size
img_size = args.img_size
ckpt_path = args.ckpt_pth
latent_dim = args.latent_dim
num_samples = args.num_samples
ckp_path = ckpt_path + dataset_name +'.pth'

figure_path = args.figure_dir + dataset_name
os.makedirs(figure_path, exist_ok=True)

''' Load testing data '''
train_dataloader, test_dataloader = load_data(dataset_name=dataset_name, normal_class=0, args=args)

''' Autoencoder model '''
model = AutoEncoder_CNN(channel=3, dim=latent_dim, large=True).to(device)
model.load_state_dict(torch.load(ckp_path))
model.eval()

normal_errors = []
total_errors = []
y_true = []
result = []

# Calculate the average reconstruction error on training data (normal)
training_error = []
for img, label in train_dataloader:
    img = img.to(device)
    output = model(img)
    training_error.append(loss_function(img, output).item())

print("\n===== Training Data =====")
avg_raw_error = np.mean(training_error)
print('Average Normal Error: {:.4f}'.format(avg_raw_error))

std_error = np.std(training_error)
avg_normal_error = avg_raw_error + 2*std_error
print('New Average Normal Error: {:.4f}, Std: {:.4f}'.format(avg_normal_error, std_error))

max_normal_error = np.max(training_error)
print('Max Normal Error: {:.4f}'.format(max_normal_error))
print("=========================\n")

result.append({
    "Average Normal Error": avg_normal_error,
    "Max Normal Error": max_normal_error,
    "Average Raw Error": avg_raw_error,
    "std_error": std_error,
})

# Evaluate on testing data
for idx, (img, label) in enumerate(tqdm(test_dataloader)):
    img = img.to(device)
    output = model(img)

    error = loss_function(img, output).item()
    total_errors.append(error)
    y_true.append(0 if label == 0 else 1)

    item = {
        "ID": idx,
        "Label": label.item(),
        "Error": error,
        "Prediction": 1 if error > avg_normal_error else 0
    }
    result.append(item)

    if idx < num_samples:
        save_image(make_grid(invTrans(torch.cat([img, output]))), "{}/{}.png".format(figure_path, idx))

y_pred_avg = [1 if error > avg_normal_error else 0 for error in total_errors]
y_pred_max = [1 if error > max_normal_error else 0 for error in total_errors]
print('Accuracy [mean]: {:.4f}'.format(accuracy_score(y_true, y_pred_avg)))
print('Accuracy [max]: {:.4f}'.format(accuracy_score(y_true, y_pred_max)))
print('Accuracy [roc]: {:.4f}'.format(roc_auc_score(y_true, total_errors)))

with open("./jsons/result.json", "w+") as f:
    json.dump(result, f, indent=4)
print('-- Save result to result.json')