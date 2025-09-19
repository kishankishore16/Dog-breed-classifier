import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from train import DogBreedClassifier  # import your model class

# Paths
MODEL_PATH = "C:\kishorecode\dog-breed-classifier\models\best_model.pth"
DATA_DIR = "C:\kishorecode\dog-breed-classifier\data\val"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DogBreedClassifier(num_classes=120)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
valid_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")
