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
from training import FruitCNN

# Setting up Flask
app = Flask(__name__)
device = torch.device("cpu")


# Get checkpoint of saved model
checkpoint = torch.load("model/model.pth", map_location = device)

# Load saved model using number of classes saved in model checkpoint
NUM_CLASSES = len(checkpoint.get('classes'))
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
raises: BadRequest: If the file format is not valid
"""
def allowed_file(file) -> bool:
    return '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    file_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Cannot Process image", "details": str(e)}), 400


MODEL_INFO = {
    "num_classes": NUM_CLASSES,
    "classes": checkpoint.get('classes'),
    "val_acc": checkpoint.get("val_acc"),
    "epoch": checkpoint.get("epoch"),
    "architecture": "FruitCNN",
    "input_size": 64,
    "dataset": "Fruits-360 64x64"
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
    return render_template("home.html")



@app.route("/predict_ui", methods=["GET"])
def predict_ui():
    return render_template("predict_ui.html")

"""
Takes user uploaded image and returns the model's predicted class in a json

returns: json file with model prediction and confidence
"""
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.mimetype.startswith("image/"):
            return jsonify({"error": "Invalid image type"}), 400

        image = Image.open(file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make a prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, pred = torch.max(prob, dim=1)

        # Return prediction
        return jsonify({
            "prediction": checkpoint["classes"][pred.item()],
            "confidence": float(confidence.item())
           
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/metrics")
def metrics():
    return render_template("metrics.html", info=MODEL_INFO)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
