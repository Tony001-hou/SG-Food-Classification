from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.datasets.folder import make_dataset
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
from cnn_model import *
from torchsummary import summary
from resnest.torch import resnest50
from d2l import torch as d2l
from ptflops import get_model_complexity_info
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18", help="name of the model")
opt = parser.parse_args()

trainlogs = logging.getLogger(__name__)
trainlogs.setLevel(logging.INFO)

logfile = logging.FileHandler("train.log")
logfile.setLevel(logging.INFO)
trainlogs.addHandler(logfile)

trainlogs.info(opt)

# Define the dataset class
class sg_food_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root, class_id, transform=None):
        self.class_id = class_id
        self.root = root
        all_classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not all_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        self.classes = [all_classes[x] for x in class_id]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = make_dataset(self.root, self.class_to_idx, extensions=('jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, "rb") as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet prior
    ]),
    'val': transforms.Compose([
        # Define data preparation operations for testing/validation set here.
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet prior
    ]),
}

data_dir = os.path.join('data', 'sg_food')
subfolder = {'train': 'train', 'val': 'val'}

# Define the dataset
selected_classes = [0,1,2,3,7]
n_classes = len(selected_classes)
image_datasets = {x: sg_food_dataset(root=os.path.join(data_dir, subfolder[x]),
                                     class_id=selected_classes,
                                     transform=data_transforms[x]) 
                  for x in ['train', 'val']}
class_names = image_datasets['train'].classes
# print('selected classes:\n    id: {}\n    name: {}'.format(selected_classes, class_names))

# Define the dataloader
batch_size = 8
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# print(f'dataset_sizes: {dataset_sizes}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'Device: {torch.cuda.get_device_name(0)}')

test_dir = os.path.join('data', 'sg_food', 'test')

# Define the test set.
test_dataset = sg_food_dataset(root=test_dir, class_id=selected_classes, transform=data_transforms['val'])
test_sizes = len(test_dataset)

# Define the dataloader for testing.
test_batch_size = 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    final_train_loss = []
    final_train_acc = []
    final_val_loss = []
    final_val_acc =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

  
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

  
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                final_train_loss.append(epoch_loss)
                final_train_acc.append(epoch_acc)
            else:
                final_val_loss.append(epoch_loss)
                final_val_acc.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


learning_rate = 1e-4
weight_decay = 1e-3
num_epoch = 20

# model = alexnet(5)
if opt.model == 'densenet169':
    model = densenet169(5)
if opt.model == 'densenet161':
    model = densenet161(5)
if opt.model == 'densenet201':
    model = densenet201(5)

if opt.model == 'shufflenet_v2_x1_5':
    model = shufflenet_v2_x1_5(5)
if opt.model == 'shufflenet_v2_x2_0':
    model = shufflenet_v2_x2_0(5)

if opt.model == 'mobilenet_v3_small':
    model = mobilenet_v3_small(5)

if opt.model == 'resnext101_32x8d':
    model = resnext101_32x8d(5)

if opt.model == 'wide_resnet101_2':
    model = wide_resnet101_2(5)

if opt.model == 'mnasnet0_5':
    model = mnasnet0_5(5)
if opt.model == 'mnasnet0_75':
    model = mnasnet0_75(5)
if opt.model == 'mnasnet1_0':
    model = mnasnet1_0(5)
if opt.model == 'mnasnet1_3':
    model = mnasnet1_3(5)

if opt.model == 'efficientnet_b0':
    model = efficientnet_b0(5)
if opt.model == 'efficientnet_b1':
    model = efficientnet_b1(5)
if opt.model == 'efficientnet_b2':
    model = efficientnet_b2(5)
if opt.model == 'efficientnet_b3':
    model = efficientnet_b3(5)
if opt.model == 'efficientnet_b4':
    model = efficientnet_b4(5)
if opt.model == 'efficientnet_b5':
    model = efficientnet_b5(5)
if opt.model == 'efficientnet_b6':
    model = efficientnet_b6(5)
if opt.model == 'efficientnet_b7':
    model = efficientnet_b7(5)

if opt.model == 'convnext_tiny':
    model = convnext_tiny(5)
if opt.model == 'convnext_small':
    model = convnext_small(5)
if opt.model == 'convnext_base':
    model = convnext_base(5)
if opt.model == 'convnext_large':
    model = convnext_large(5)

if opt.model == 'resnest50':
    model = convnext_tiny(5)
if opt.model == 'resnest101':
    model = convnext_small(5)
if opt.model == 'resnest200':
    model = convnext_base(5)
if opt.model == 'resnest269':
    model = convnext_large(5)

model = model.to(device)
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

trainlogs.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
trainlogs.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

model = train_model(model, criterion, optimizer, scheduler, num_epochs = 20)

model.eval()

test_acc = 0

print('Evaluation')
print('-' * 10)

with torch.no_grad():
    # Iterate over the testing dataset.
    for (inputs, labels) in test_loader:
        inputs = inputs.to(device)
        # Predict on the test set
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
        test_acc += torch.sum(preds == labels.data)

# Compute the testing accuracy
test_acc = test_acc.double() / test_sizes
print('Testing Acc: {:.4f}'.format(test_acc))

trainlogs.info('Testing Acc: {:.4f}'.format(test_acc))