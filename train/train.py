import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from fire import Fire

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
# from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from torchmetrics.classification import Accuracy

from dataset import GestureDataset
from model import get_model
from utils.file_utils import load_config
from utils.to_utils import DictToObject

torch.set_float32_matmul_precision("medium")

def get_use_labels_id(labels: list[str], use_labels: list[str]) -> list[int]:
    use_labels_id = []
    for v_label in use_labels:
        for i, label in enumerate(labels):
            if v_label == label:
                use_labels_id.append(i)
                break
    return use_labels_id

class GestureDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        use_labels_id = get_use_labels_id(config.labels, config.use_labels)
        self.batch_size = config.batch_size
        self.train_dataset = GestureDataset(config.train_x_files, config.train_y_files, use_labels_id)
        self.valid_dataset = GestureDataset(config.valid_x_files, config.valid_y_files, use_labels_id)
    
    def setup(self, stage):
        ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, persistent_workers=True)

class GestureModule(LightningModule):
    def __init__(self, config, weight: torch.Tensor):
        super().__init__()
        if config.precision == 16:
            weight = weight.to(dtype=torch.float16)
        self.config = config
        self.model = get_model(config)
        self.lr = config.lr
        self.accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        if config.balance_samples:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)

        self.log(f'train loss', loss, prog_bar=True)
        self.log(f'train accuracy', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)

        self.log(f'valid loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'valid accuracy', acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        eps = 1e-5 if self.config.precision == 16 else 1e-8
        print('eps', eps)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        return optimizer

def setdefault(config, attr, value) -> None:
    if not hasattr(config, attr):
        setattr(config, attr, value)

def train(config: str):
    config_path = config
    config = DictToObject(load_config(config))
    setattr(config, 'num_classes', len(config.use_labels))
    setdefault(config, 'precision', 32)
    data_module = GestureDataModule(config)
    gesture_module = GestureModule(config, weight=data_module.train_dataset.weight)
    logger = TensorBoardLogger("tb_logs", name="gesture", version=config_path.strip().split('/')[-1].split('.')[0])
    # print('16-true' if config.precision == 16 else '32-true')
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        # callbacks=[ ModelCheckpoint(every_n_epochs=10, save_top_k=3), ],
        default_root_dir='./log',
        logger=logger,
        precision='16-true' if config.precision == 16 else '32-true',
    )
    tuner = Tuner(trainer)
    # print('lr: ', config.lr)
    # print('batch_size: ', config.batch_size)
    trainer.fit(model=gesture_module, datamodule=data_module)

if __name__ == '__main__':
    Fire(train)
