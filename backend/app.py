import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1 make a tranfom that will be used to turn the images into vectores and it will resize,convert to tensor, then normalize

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4688, 0.4635, 0.3434],
        std=[0.2033, 0.2051, 0.1967]
    )
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pulls in the images and uses the prior tansform to turn the images into vectors

train_data = datasets.ImageFolder(root='backend/data/train', transform=transform)
val_data = datasets.ImageFolder(root='backend/data/validation', transform=transform)
test_data = datasets.ImageFolder(root='backend/data/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3 Sets up device to use the gpu if avalible and cuda then seeding if gpu is available so the resaults stay the same

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4 Creating the model itself with what each layer does and how

class VegetableCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(VegetableCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy_out = self._forward_conv(dummy)
            self.flatten_dim = dummy_out.numel()

        self.fc1 = nn.Linear(self.flatten_dim, 256)

        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        
    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5 pulls model down and puts it in the cpu/gpu and determine what algorithm to use

model = VegetableCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6 detirmine how many loops you run through and pulling data and putting it into the model to train

num_epochs = 10

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