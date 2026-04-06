# src/model.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torchmetrics
from src import config
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class LitModel(pl.LightningModule):

    def __init__(self, model_name):
        super().__init__()

        self.save_hyperparameters()

        num_classes = config.NUM_CLASSES

        if num_classes is None:
            raise ValueError("NUM_CLASSES not set")

     
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif model_name == "mobilenet":
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, num_classes
            )

        else:
            raise ValueError("Unknown model")

        self.criterion = nn.CrossEntropyLoss()

        # 🔥 separate metrics (important)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    # =========================
    # TRAINING
    # =========================
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)


    def configure_optimizers(self):

        if config.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )

        elif config.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=config.WEIGHT_DECAY
            )

        else:
            raise ValueError("Unknown optimizer")

        if config.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.STEP_SIZE,
                gamma=config.GAMMA
            )

            return [optimizer], [scheduler]

        return optimizer