# utils.py

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model(model, checkpoint_path):
    """
    Load trained model weights
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def preprocess_image(image_path, transform):
    """
    Load and preprocess single image
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(model, image_tensor, class_names):
    """
    Predict single image
    """
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return class_names[pred], probs.cpu().numpy()

def get_latest_checkpoint(model_dir="models"):
    files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
    files.sort()
    return os.path.join(model_dir, files[-1]) if files else None
    
def plot_confusion_matrix(cm, class_names):
    """
    Simple confusion matrix plot
    """
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()