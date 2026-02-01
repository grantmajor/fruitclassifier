import torch
from training import FruitCNN  # your CNN class
from torchvision import transforms
from PIL import Image

# Load checkpoint
checkpoint = torch.load("model/model.pth", map_location="cpu")
classes = checkpoint["classes"]  # this is trainset.classes
num_classes = len(classes)

# Load model
model = FruitCNN(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use the same test_transform as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Pick a known validation image (replace with a path that exists)
img_path = "model/r1_75_100.jpg"
image = Image.open(img_path).convert("RGB")
x = transform(image).unsqueeze(0)  # add batch dimension

# Predict
with torch.no_grad():
    out = model(x)
    pred_index = out.argmax(dim=1).item()
    pred_class = classes[pred_index]

print("Predicted index:", pred_index)
print("Predicted class:", pred_class)
