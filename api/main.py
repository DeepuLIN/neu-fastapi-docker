from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from PIL import Image
import torch
import io
import numpy as np
import cv2
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src import config
from src.model import LitModel  



app = FastAPI()
DEVICE = torch.device("cpu")

MODEL_PATH = "models/mobilenet-best.ckpt"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found at:", MODEL_PATH)
        return None

    print("🚀 Loading model...")
    model = LitModel.load_from_checkpoint(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print("✅ Model loaded successfully on CPU")
    return model


model = load_model()


if model is not None:
    try:
        if hasattr(model, "model"):
            backbone = model.model
            print("✅ Using inner model from Lightning wrapper")
        else:
            backbone = model

        target_layer = backbone.features[-1]
        print("✅ Using MobileNet Grad-CAM layer")

        cam = GradCAM(model=backbone, target_layers=[target_layer])

    except Exception as e:
        print(f"❌ Grad-CAM setup failed: {e}")
        backbone = None
        cam = None
else:
    print("⚠️ Model not loaded — skipping Grad-CAM")
    backbone = None
    cam = None


class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "NEU Surface Defect Classifier API 🚀"}


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "gradcam_ready": cam is not None
    }


@app.post("/predict/image", response_class=StreamingResponse)
async def predict_image(file: UploadFile = File(...)):

    if model is None or cam is None:
        return {"error": "Model not loaded"}

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform
        input_tensor = config.VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = float(torch.max(probs))

        # Prepare for Grad-CAM
        h, w = input_tensor.shape[-2:]
        image_np = np.array(image.resize((w, h))) / 255.0

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

        # Label
        label = f"{class_names[pred]}: {confidence:.2f}"

        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.rectangle(visualization, (0, 0), (w, 40), (0, 0, 0), -1)

        cv2.putText(
            visualization,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Encode output
        _, buffer = cv2.imencode(".jpg", visualization)

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:
        return {"error": str(e)}