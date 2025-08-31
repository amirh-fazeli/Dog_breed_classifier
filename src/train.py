# src/train.py
import os
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

DATA_ROOT = Path('./data/10breeds')
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 12
LR = 1e-3
MODEL_OUT = "../best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_dataloaders(data_root, batch_size=BATCH_SIZE):
    train_ds = datasets.ImageFolder(root=str(data_root / 'train'), transform=train_transforms)
    val_ds = datasets.ImageFolder(root=str(data_root / 'val'), transform=val_transforms)
    test_ds = datasets.ImageFolder(root=str(data_root / 'test'), transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_ds.classes

def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    # freeze base layers optionally
    for param in model.parameters():
        param.requires_grad = True  # set False to freeze
    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_ROOT)
    model = build_model(len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        print(f" Train loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

        # val
        model.eval()
        val_running_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs,1)
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
        val_acc = val_running_corrects / val_total
        print(f" Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'model_state_dict': model.state_dict(),
                        'class_names': class_names}, MODEL_OUT)
            print(f" Saved best model (val_acc={best_acc:.4f}) -> {MODEL_OUT}")

        scheduler.step()

    print(f"Training complete. Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()
