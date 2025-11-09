import os
import torch
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