import torch
import time
import os
import torchvision.transforms as transforms
import seaborn as sns
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict
from datetime import datetime

# Define the CNN
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        self.fc_dropout = nn.Dropout(0.5)

        # We use Adaptive pooling to lower output size
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x

"""
Tracks the percentage of times that the ground truth is within the top k predicted labels

:param outputs: predicted output from model
:param labels: ground truth labels
:param k: number of predicted labels to check for the ground truth label
"""
def top_k_accuracy(outputs, labels, k=5):
    with torch.no_grad():
        _, top_k = outputs.topk(k, dim=1)
        correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
        return correct.any(dim=1).float().mean().item()

def main():
    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f'Computing device: {device}')
    METRICS_DIR = "static/metrics"
    os.makedirs(METRICS_DIR, exist_ok=True)

    PLOTS_DIR = os.path.join(METRICS_DIR, 'plots')
    TABLES_DIR = os.path.join(METRICS_DIR, 'tables')


    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Data augmentation on training data
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),  # simulates different fruit orientations
        transforms.RandomRotation(10),  # simulates different camera angles
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),  # simulates lighting condition variability
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

    trainset = datasets.ImageFolder(
        root='data/Training',
        transform=train_transform
    )  # pull training set and apply transformations

    testset = datasets.ImageFolder(
        root='data/Test',
        transform=test_transform
    )  # pull test set and apply transformations

    valset = datasets.ImageFolder(
        root='data/Validation',
        transform=test_transform
    )  # pull validation set and apply transformations


    batch_size = 64
    train_loader = DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=6,
        pin_memory=True
    )  # create dataloader for training set

    test_loader = DataLoader(
        testset, batch_size=batch_size,
        shuffle=False, num_workers=6
    )  # create dataloader for test set

    val_loader = DataLoader(
        valset, batch_size=batch_size,
        shuffle=False, num_workers=6
    )

    # Create model object
    num_classes = len(trainset.classes)
    model = FruitCNN(num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training metrics
    num_epochs = 50
    best_val_acc, best_val_loss = 0.0, float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    top_k_accs = []

    # Scheduler to hasten convergence (hopefully)
    scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs)

    # 6 consecutive epochs without improvement stops training
    patience = 6
    epoch_no_improve = 0
    eps  = 1e-6 #threshold used for fine-point floating point operations

    # Time training
    start_time = time.time()
    best_epoch = 0

    # Pretty standard CNN training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()


        train_loss = running_loss / total
        train_acc = correct / total

        # Model validation
        model.eval()

        running_loss, val_topk = 0.0, 0.0
        correct, total = 0, 0
        if len(val_loader.dataset) == 0:
            raise RuntimeError("Validation dataset is empty! Check folder paths and contents.")

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

                # Top-K accuracy
                val_topk += top_k_accuracy(outputs, labels, k=5) * labels.size(0)

                total += labels.size(0)


        val_loss = running_loss / total
        val_acc = correct / total
        val_topk /= total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        top_k_accs.append(val_topk)



        # Output epoch-related metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]"
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Val Top-K: {val_topk:.4f}")

        # Save model, we use an epsilon variable to handle floating point errors at small scales
        if (val_acc > best_val_acc + eps) or (val_loss - best_val_loss <= eps and val_loss +eps < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'classes': trainset.classes,
                'epoch': epoch,
                'val_acc': val_acc
            }, 'model/model.pth')
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
        # Stop training when val accuracy does not improve
        if epoch_no_improve >= patience:
            print("Patience limit reached, training stopped.")
            break

    metrics_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "top_k_acc": top_k_accs
    })

    metrics_df.to_csv(
        os.path.join(TABLES_DIR, "training_metrics.csv"),
        index=False
    )

    # Stop timer and calculate training time per epoch
    total_time = time.time() - start_time
    avg_epoch_time = total_time / (epoch+1)

    print(f"Best Epoch: {best_epoch}")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average training timer per epoch: {avg_epoch_time:.2f} seconds")

    # Confusion matrix code
    checkpoint = torch.load('model/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_preds, all_labels = [], []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Collect predictions for class accuracy table.
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

        # Create CM
        cm = confusion_matrix(all_labels, all_preds)

        # Normalize CM to increase interpretability in the case of unbalanced classes
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Compute macro F1
        macro_f1 = f1_score(all_labels, all_preds, average='macro')


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_summary = {
        "Total Epochs": len(train_losses),
        "Best Epoch": best_epoch,
        "Best Validation Accuracy": best_val_acc,
        "Macro-F1": macro_f1,
        "Number of Parameters": num_params,
        "Total Training Time (seconds)": round(total_time, 2),
        "Training Time per Epoch (seconds)": round(avg_epoch_time, 2)
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"train_summary_{timestamp}.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    class_names = testset.classes
    rows = []

    for idx, class_name in enumerate(class_names):
        total = class_total[idx]
        correct = class_correct[idx]
        acc = correct / total if total > 0 else 0.0

        rows.append({
            "Class": class_name,
            "Accuracy": round(acc, 4),
            "Samples": total
        })

    df = pd.DataFrame(rows).sort_values("Accuracy")
    df.to_csv(os.path.join(TABLES_DIR, "per_class_accuracy.csv"),
              index=False)

    # Confusion Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        cbar = True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized) - Test Set")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "confusion_matrix.png"),
        dpi=300
    )
    plt.close()


    # Loss visualization
    epochs = range(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "loss_curve.png"),
        dpi=300
    )
    plt.close()

    # Accuracy visualization
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "accuracy_curve.png"),
        dpi=300
    )
    plt.close()




if __name__ == "__main__":
    main()