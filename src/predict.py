# predict.py
import torch
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# ----- Load model -----
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint['class_names']

model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_breed(image):
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_class = class_names[pred_idx]
        # top 3 as dict
        top3_idx = probs.argsort()[-3:][::-1]
        top3_dict = {class_names[i]: float(probs[i]) for i in top3_idx}
    return pred_class, top3_dict

