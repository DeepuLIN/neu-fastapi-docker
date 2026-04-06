# src/inference.py

import torch
from PIL import Image
import config
from model import LitModel
import os

def load_model():
    model = LitModel.load_from_checkpoint(config.CHECKPOINT_PATH)
    model = model.to(config.DEVICE)
    model.eval()
    return model


def predict(image_path, model, transform, class_names):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return class_names[pred], probs.cpu().numpy()


if __name__ == "__main__":

    model = load_model()

    test_dir = "tests"

    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled_in_scale",
        "scratches"
    ]

    print("\n🔍 Running inference on test images...\n")

    for img_name in os.listdir(test_dir):

        # skip non-images
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(test_dir, img_name)

        label, _ = predict(img_path, model, config.VAL_TRANSFORM, class_names)

        print(f"{img_name} → {label}")