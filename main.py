import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

MODEL_PATH = os.path.join("models", "tb_pneumonia_model.pth")
VAL_DIR = "train"  
IMAGE_PATH = "images/TB-2.jpeg"  # Image 
BATCH_SIZE = 16
LABELS = {0: "Normal", 1: "Tuberculosis", 2: "Pneumonia"}

# Model Setup
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load 
def load_image(path):
    image = Image.open(path).convert("RGB")
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input X-ray Image")
    plt.show()
    return transform(image).unsqueeze(0)

def predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return LABELS[predicted.item()]

# Generate Description
def generate_description(disease_label):
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"This is a chest X-ray showing signs of {disease_label}. The possible medical analysis includes:"
    description = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return description

# Model
def evaluate_model():
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # To find accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc:.4f}")

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=val_dataset.classes,
                yticklabels=val_dataset.classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def analyze_xray(image_path):
    img_tensor = load_image(image_path)
    label = predict(img_tensor)
    print(f"\n Detected Disease: {label}")

    description = generate_description(label)
    print(f"\n Descriptive Analysis:\n{description}")

if __name__ == "__main__":
    analyze_xray(IMAGE_PATH)
    evaluate_model()
