import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image

#TODO: Check the classes.json, it was saved separate from training

# CNN from training.py
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

# Setting up Flask
app = Flask(__name__)
device = torch.device("cpu")

# Pull classes from .json
with open("model/classes.json") as f:
    CLASSES = json.load(f)

num_classes = len(CLASSES)

# Load saved model
model = FruitCNN(num_classes)
checkpoint = torch.load("model/model.pth", map_location = device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Use same transforms as test_transform from training.py
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No Image Provided"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs,1)

    return jsonify({
        "class": CLASSES[pred_idx.item()],
        "confidence": round(confidence.item(),4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
