import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os

# Paths
MODEL_PATH = "models/best_model.pth"

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")
MODEL_PATH_ABS = os.path.join(PROJECT_ROOT, "..", MODEL_PATH)

# You should get the class names from the training dataset directory
# This assumes the data directory structure is `data/train/class_name_folders`
CLASS_NAMES = sorted([d.name for d in os.scandir(os.path.join(DATA_DIR, "train")) if d.is_dir()])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
NUM_CLASSES = 120
# Define the same model architecture as in train.py
model = models.resnet50()
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, NUM_CLASSES)

try:
    model.load_state_dict(torch.load(MODEL_PATH_ABS, map_location=device))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH_ABS}. Please run train.py first.")
    exit()

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    # Use a try-except block to handle file not found errors gracefully
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"Error: Image file not found at '{image_path}'."

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)

    if CLASS_NAMES:
        return f"Predicted Breed: {CLASS_NAMES[predicted_idx.item()]}"
    else:
        return f"Predicted class index: {predicted_idx.item()}"

# Example with fixed path using raw string
image_to_predict = r"C:\kishorecode\dog-breed-classifier\data\val\n02085620-Chihuahua\n02085620_11477.jpg"
print(predict(image_to_predict))
