# -*- coding: utf-8 -*-
"""dl.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Xg46U3Pb42e2N5WTQCK2w8PcZjZbcf4U
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Function to load CIFAR-10 dataset from binary pickle files
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Define paths to the data files
file_paths = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

unlabeled_file_path = 'cifar_test_nolabels.pkl'

# Detect and set the appropriate device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load and prepare the training data
batch_data = []
batch_label = []
for file_path in file_paths:
    batch = unpickle(file_path)
    batch_data.append(batch[b'data'])
    batch_label.append(batch[b'labels'])
train_data = np.concatenate(batch_data)
train_labels = np.concatenate(batch_label)
print(train_data.shape, train_labels.shape)

# Load and prepare the test data
test_data_dict = unpickle('test_batch')
test_data = np.array(test_data_dict[b'data'])
test_labels = np.array(test_data_dict[b'labels'])

# Load and prepare unlabeled data for semi-supervised learning
unlabeled_data_dict = unpickle(unlabeled_file_path)
unlabeled_data = np.array(unlabeled_data_dict[b'data'])

# Define the dataset class for CIFAR-10
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        return image, -1  # Return -1 for unlabeled data

# Data transformations for training and testing
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Create dataset objects
train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

# Data loaders for handling batches
test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

unlabeled_dataset = CIFAR10Dataset(unlabeled_data, transform=transform_test)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32  # Reduced from 64 to adjust to CIFAR-10
        # Additional layers and network initialization
        # Implement the network forward pass

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 128, max(1, num_blocks[2]-1), stride=2)  # 减少一个block
        self.linear = nn.Linear(512, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
          layers.append(block(self.in_planes, planes, stride))
          self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet34():
  return ResNet(BasicBlock,[3,4,6,3])
# Instantiate and train the network
model = resnet34().to(device)

def train(model, data_loader, optimizer, criterion, device, lr_scheduler=None):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        _, preds = torch.max(outputs, 1)
        epoch_acc += torch.sum(preds == targets).item()
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(data_loader)
    average_accuracy = epoch_acc / (len(data_loader) * data_loader.batch_size)

    return average_loss, average_accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                epoch_acc += torch.sum(preds == targets).item()
                epoch_loss += loss.item()
                total_samples += targets.size(0)
            elif len(batch) == 1:  # Handling data loaders that only return inputs (for predictions)
                inputs = batch[0].to(device)
                outputs = model(inputs)
                # Optionally handle outputs if needed (for example logging or further processing)

    if total_samples > 0:
        average_loss = epoch_loss / len(data_loader)
        average_accuracy = epoch_acc / total_samples
    else:
        average_loss = None  # No loss computed if there are no targets
        average_accuracy = None  # No accuracy computed if there are no targets

    return average_loss, average_accuracy

import torchvision
from torchsummary import summary
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import json
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from torchvision.transforms import autoaugment
EPOCHS=35
model = resnet34().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
lr_scheduler = OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, div_factor=10, pct_start=0.3)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


for epoch in range(EPOCHS):
    try:
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, lr_scheduler)
    except Exception as e:
        print(f"Training failed during epoch {epoch + 1}: {e}")
        train_loss, train_acc = None, None

    try:
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    except Exception as e:
        print(f"Evaluation failed during epoch {epoch + 1}: {e}")
        val_loss, val_acc = None, None

    train_loss_str = f"{train_loss:.3f}" if train_loss is not None else "N/A"
    train_acc_str = f"{train_acc:.2f}%" if train_acc is not None else "N/A"
    val_loss_str = f"{val_loss:.3f}" if val_loss is not None else "N/A"
    val_acc_str = f"{val_acc:.2f}%" if val_acc is not None else "N/A"

    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss_str}, Training Accuracy: {train_acc_str}, Validation Loss: {val_loss_str}, Validation Accuracy: {val_acc_str}')

    if val_acc is not None and val_acc == max(val_accuracies, default=float('-inf')):
        torch.save(model.state_dict(), 'best_model3.pth')
        print("Model saved at epoch:", epoch + 1)

import time
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

model.load_state_dict(torch.load('best_model3.pth'))
model = model.to(device)
model.eval()

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"The final test accuracy is: {test_acc*100:.2f}% and the test loss is: {test_loss:.3f}")

with open('cifar_test_nolabels.pkl', 'rb') as file:
    test_data_dict = pickle.load(file)

class CIFARTestDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict[b'data']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        if self.transform:
            image = self.transform(image)
        return image

# Initialize Dataset and DataLoader
test_dataset = CIFARTestDataset(data_dict=test_data_dict, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

def predict(model, test_loader):
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Get predictions
predictions = predict(model, test_loader)

# Prepare and save the CSV file
ids = list(range(10000))
df = pd.DataFrame({'ID': ids, 'Labels': predictions})
df.to_csv('submission_3.csv', index=False)

