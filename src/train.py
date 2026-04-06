# src/train.py

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import get_dataloaders
from model import LitModel
import config


def train_model(model_name):

    train_loader, val_loader, classes = get_dataloaders(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )

    config.NUM_CLASSES = len(classes)

    print(f"\n🚀 Training {model_name}")
    print(f"📊 Classes: {config.NUM_CLASSES}")


    model = LitModel(model_name=model_name)

    mlf_logger = MLFlowLogger(
            experiment_name="neu-classification",
            tracking_uri="sqlite:///mlflow.db"
    )

   
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=config.MODEL_SAVE_DIR,
        filename=f"{model_name}-best"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        min_delta=0.001
    )

  
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="gpu" if config.DEVICE == "cuda" else "cpu",
        devices=1,
        precision=config.PRECISION if config.DEVICE == "cuda" else 32,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,

   
        check_val_every_n_epoch=1,

     
        benchmark=True
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"✅ Finished training {model_name}")


if __name__ == "__main__":

    for model_name in config.MODELS:
        train_model(model_name)