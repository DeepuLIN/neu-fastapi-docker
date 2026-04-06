# gradcam.py

import torch
import numpy as np
import cv2
import os
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.model import LitModel
from torchvision.models import mobilenet_v2

from src import config


def load_model():

    model_path = config.BEST_MODEL

    model = LitModel.load_from_checkpoint(model_path)

    model = model.to(config.DEVICE)
    model.eval()

    
    return model


def get_target_layer(model):
    return model.model.features[-1]



def generate_gradcam(image_path):

    model = load_model()

    target_layer = get_target_layer(model)

    cam = GradCAM(model=model, target_layers=[target_layer])


    image = Image.open(image_path).convert("RGB")

    input_tensor = config.VAL_TRANSFORM(image)
    input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
    
    h, w = input_tensor.shape[-2:]
    image_np = np.array(image.resize((w, h))) / 255.0

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)


    output_path = image_path.replace(".jpg", "_gradcam.jpg")
    cv2.imwrite(output_path, visualization)

    print(f"Grad-CAM saved at: {output_path}")


if __name__ == "__main__":
    generate_gradcam("tests/1.jpg")