import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os # Import the os module

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
TRAIN_DIR = "data/train"
VALID_DIR = "data/val"
MODEL_PATH = "models/best_model.pth"

# Hyperparameters
NUM_CLASSES = 120
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets & Loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model (using new weights API)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Replace final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- FIX for RuntimeError: Parent directory does not exist ---
# Create the parent directory for the model path if it doesn't exist.
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Training loop
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        torch.save(model.state_dict(), MODEL_PATH)
        best_acc = acc
        print(f"✅ Saved Best Model (Val Acc: {best_acc:.2f}%)")

print("Training complete. Best Accuracy:", best_acc)
torch.save(model.state_dict(), "best_model.pth")
