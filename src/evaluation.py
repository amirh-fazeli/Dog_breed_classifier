# src/evaluate.py
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Paths and device
DATA_ROOT = Path('./data/10breeds')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../best_model.pth"
BATCH_SIZE = 32

# Transforms for test/validation
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load test dataset
test_ds = datasets.ImageFolder(root=str(DATA_ROOT / 'test'), transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = models.resnet18(pretrained=False)  # architecture must match
num_classes = len(checkpoint['class_names'])
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Overall accuracy
accuracy = np.mean(all_preds == all_labels)
print(f"Test set accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Per-class accuracy / classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=checkpoint['class_names']))
