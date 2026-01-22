from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

model = torch.load("model/model.pth", map_location="cpu")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



