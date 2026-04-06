# src/evaluate.py

import torch
from sklearn.metrics import confusion_matrix, classification_report


from dataset import get_dataloaders
from model import LitModel
from utils import plot_confusion_matrix
import config


def evaluate_model():

    train_loader, val_loader, classes = get_dataloaders(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )

    config.NUM_CLASSES = len(classes)


    model = LitModel.load_from_checkpoint(config.CHECKPOINT_PATH)
    model = model.to(config.DEVICE)
    model.eval()

 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds)

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    plot_confusion_matrix(cm, classes)


if __name__ == "__main__":
    evaluate_model()