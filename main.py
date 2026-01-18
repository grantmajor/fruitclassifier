import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# Use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computing device: {device}')

# Data augmentation on training data
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5), # simulates different fruit orientations
    transforms.RandomRotation(10),          # simulates different camera angles
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),                                      # simulates lighting condition variability
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]

    )
])

# Transforms on testing data
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

batch_size = 16     # batch size set to 16 due to hardware constraints
num_epochs = 30

trainset = datasets.ImageFolder(
    root='data/Training',
    transform=train_transform
)                   # pull training set and apply transformations

testset = datasets.ImageFolder(
    root='data/Test',
    transform=test_transform
)                   # pull test set and apply transformations

valset = datasets.ImageFolder(
    root='data/Validation',
    transform=test_transform
                    # pull validation set and apply transformations
)



train_loader = DataLoader(
    trainset, batch_size=batch_size,
    shuffle=True, num_workers=4
)                   # create dataloader for training set

test_loader = DataLoader(
    testset, batch_size=batch_size,
    shuffle=False, num_workers=4
)                   # create dataloader for test set

val_loader = DataLoader(
    valset, batch_size=batch_size,
    shuffle=False, num_workers=4
)
images, labels = next(iter(train_loader))



# Define the CNN
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model object
num_classes = len(trainset.classes)
model = FruitCNN(num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Network training

# Training metrics
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train() # Not needed until dropout or batchnorm layers are used
    running_loss, correct, total = 0.0, 0, 0

    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        inputs, labels = data

        # Forward pass, backward pass, optimization
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # Print metrics
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)


    train_loss = running_loss / total
    train_acc = correct / total

    # Model validation
    model.eval() # Not needed unless dropout or batchnorm layers are used
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]"
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, 'model.pth')

# Visualize NN Metrics

# Loss visualization
epochs = range(1, num_epochs+1)
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy visualization
plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


