import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision as models

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1 make a tranfom that will be used to turn the images into vectores and it will resize,convert to tensor, then normalize

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pulls in the images and uses the prior tansform to turn the images into vectors

train_data = datasets.ImageFolder(root='backend/data/train', transform=transform)
val_data = datasets.ImageFolder(root='backend/data/validation', transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3 Sets up device to use the gpu if avalible and cuda then seeding if gpu is available so the resaults stay the same

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4 Creating the model itself with what each layer does and how

class VegetableCNN(nn.Module):
    def __init__(self, num_classes=44, conv_dropout_p=0.0, fc_dropout_p=0.5):
        super(VegetableCNN, self).__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Linear(self.backbone.fc.in_features, num_classes)

    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm2d(32)

    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm2d(64)

    #     self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    #     self.bn3 = nn.BatchNorm2d(128)

    #     self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    #     self.bn4 = nn.BatchNorm2d(256)

    #     self.conv_dropout = nn.Dropout2d(p=conv_dropout_p)

    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    #     with torch.no_grad():
    #         dummy = torch.zeros(1, 3, 224, 224)
    #         dummy_out = self._forward_conv(dummy)
    #         self.flatten_dim = dummy_out.numel()

    #     self.fc1 = nn.Linear(self.flatten_dim, 512)
    #     self.fc_bn1 = nn.BatchNorm1d(512)

    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc_bn2 = nn.BatchNorm1d(256)

    #     self.fc_out = nn.Linear(256, num_classes)

    #     self.fc_dropout = nn.Dropout(fc_dropout_p)
        
    # def _forward_conv(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = F.relu(x)
    #     x = self.pool(x)

    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = F.relu(x)
    #     x = self.pool(x)

    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = F.relu(x)
    #     x = self.conv_dropout(x)
    #     x = self.pool(x)

    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = F.relu(x)
    #     x = self.conv_dropout(x)
    #     x = self.pool(x)

    #     return x
    
    # def forward(self, x):
    #     x = self._forward_conv(x)

    #     x = x.view(x.size(0), -1)

    #     x = self.fc1(x)
    #     x = self.fc_bn1(x)
    #     x = F.relu(x)
    #     x = self.fc_dropout(x)

    #     x = self.fc2(x)
    #     x = self.fc_bn2(x)
    #     x = F.relu(x)
    #     x = self.fc_dropout(x)

    #     x = self.fc_out(x)
    #     return self.backbone(x)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5 pulls model down and puts it in the cpu/gpu and determine what algorithm to use

model = VegetableCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6 detirmine how many loops you run through and pulling data and putting it into the model to train

num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 7 takes resaults from training and testing to see how accurate the model is using data the model hasn't seen

model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():
    for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        val_total += labels.size(0)

        val_correct += (predicted == labels).sum().item()

val_loss /= len(val_loader)

val_acc = 100 * val_correct / val_total

print(
    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8 Saveing trained data

# torch.save(model.state_dict(), "fruit_vegetable_model.pth")
