# dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config


def get_dataloaders(data_dir, batch_size=32):

    

    full_dataset = datasets.ImageFolder(
        root=config.DATA_DIR,
    )

    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    train_dataset.dataset.transform = config.TRAIN_TRANSFORM
    val_dataset.dataset.transform = config.VAL_TRANSFORM

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, full_dataset.classes