import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from flask import Flask, request, jsonify, flash, redirect, url_for, Response, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io

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


# Get checkpoint of saved model
checkpoint = torch.load("model/model.pth", map_location = device)

#TODO: Get class count from the model checkpoint
NUM_CLASSES = 131
# Load saved model using number of classes saved in model checkpoint
model = FruitCNN(num_classes=NUM_CLASSES).to(device)
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

ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}


"""
Verifies if a file is a supported image format

param: filename: name of the file that is being verified
returns: boolean value stating if the file is a valid format
raises:  BadRequest: If the file format is not valid
"""
def allowed_file(filename) -> tuple[Response, int] | bool:

    # Check for corrupted files
    try:
        img = Image.open(filename)
        img.verify()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # CHeck for wrong filetype
    if not filename.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400


    # Return validity of file
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS



#TODO: Link real model data to model_info. Using fake data now.
MODEL_INFO = {
    "num_classes": NUM_CLASSES,
    "classes": NUM_CLASSES,
    "val_acc": checkpoint.get("val_acc"),
    "epoch": checkpoint.get("epoch"),
    "architecture": "FruitCNN",
    "input_size": 64
}

"""
Gets json with relevant model information

return: json file that stores model training information and metrics
"""
@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify(MODEL_INFO)


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    return render_template("predict.html")

"""
Takes user uploaded image and returns the model's predicted class in a json

returns: json file with model prediction and confidence
"""
def predict():
    if request.method =='POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.mimetype.startswith("image/"):
            return jsonify({"error": "Invalid image type"}), 400
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, pred = torch.max(prob, dim=1)
        return jsonify({
            # TODO: Add real prediction when classes are synced  CLASSES[pred.item()],
            "prediction": NUM_CLASSES,
            "confidence": float(confidence.item())
        })

@app.route("/metrics")
def metrics():
    return render_template("metrics.html", info=MODEL_INFO)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
