import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm as tdqm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4688, 0.4635, 0.3434],
        std=[0.2033, 0.2051, 0.1967]
    )
])

train_data = datasets.ImageFolder(root='backend/data/train', transform=transform)
val_data = datasets.ImageFolder(root='backend/data/validation', transform=transform)
test_data = datasets.ImageFolder(root='backend/data/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class VegetableCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(VegetableCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        
