import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision.io import decode_image
from torchvision import datasets
from torch.utils.data import DataLoader


# Using CPU-only PyTorch installation for development
    # Use GPU if possible
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Computing device: {device}')

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

trainset = datasets.ImageFolder(
    root='data/fruits-360/Training',
    transform=train_transform
)                   # pull training set and apply transformations

testset = datasets.ImageFolder(
    root='data/fruits-360/Test',
    transform=test_transform
)                   # pull test set and apply transformations

train_loader = DataLoader(
    trainset, batch_size=batch_size,
    shuffle=True, num_workers=4
)                   # create dataloader for training set

test_loader = DataLoader(
    testset, batch_size=batch_size,
    shuffle=True, num_workers=4
)                   # create dataloader for test set


images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)
print(labels[:5])
