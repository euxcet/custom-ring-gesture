import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from fire import Fire

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix

from model import get_model, use_pretrained_model
from dataset.exp_baseline_dataset import ExpBaselineDataset
from utils.config import ExpBaselineTrainConfig

# torch.set_float32_matmul_precision("medium")

def get_labels_id(labels: list[str], use_labels: list[str]) -> list[int]:
    return [labels.index(label) for label in use_labels]

class ExpBaselineDataModule(LightningDataModule):
    def __init__(self, config: ExpBaselineTrainConfig):
        super().__init__()
        custom_labels_id = get_labels_id(config.labels, config.custom_labels)
        self.config = config
        self.train_dataset = ExpBaselineDataset(
            dataset_type='train',
            x_files=config.train_x_files,
            y_files=config.train_y_files,
            custom_labels_id=custom_labels_id,
            custom_num_samples=config.custom_num_samples,
            do_aug=config.do_aug,
        )
        self.valid_dataset = ExpBaselineDataset(
            dataset_type='valid',
            x_files=config.valid_x_files,
            y_files=config.valid_y_files,
            custom_labels_id=custom_labels_id,
        )
    
    def setup(self, stage):
        ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=8, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.batch_size, num_workers=8, shuffle=False, persistent_workers=True)

class ExpBaselineModule(LightningModule):
    def __init__(self, config: ExpBaselineTrainConfig, weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.model = get_model(config)
        if config.use_pretrained_model:
            self.model = use_pretrained_model(self.model, config)
        self.accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.train_micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="micro")
        self.train_macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="micro")
        self.valid_macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        if config.balance_samples:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.num_custom_labels = len(config.custom_labels)
        
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)
        micro_f1 = self.train_micro_f1(output, y)
        macro_f1 = self.train_macro_f1(output, y)

        self.log(f'train loss', loss, prog_bar=True)
        self.log(f'train accuracy', acc, prog_bar=True, sync_dist=True)
        self.log(f'train micro f1', micro_f1, prog_bar=False, sync_dist=True)
        self.log(f'train macro f1', macro_f1, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)
        micro_f1 = self.valid_micro_f1(output, y)
        macro_f1 = self.valid_macro_f1(output, y)

        self.log(f'valid loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'valid accuracy', acc, prog_bar=True, sync_dist=True)
        self.log(f'valid micro f1', micro_f1, prog_bar=False, sync_dist=True)
        self.log(f'valid macro f1', macro_f1, prog_bar=False, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.valid_micro_f1.reset()
        self.valid_macro_f1.reset()
    
    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr, eps=self.config.eps)
        return optimizer

def setdefault(config, attr, value) -> None:
    if not hasattr(config, attr):
        setattr(config, attr, value)

def train(config: str):
    config: ExpBaselineTrainConfig = ExpBaselineTrainConfig.from_yaml(config)
    data_module = ExpBaselineDataModule(config)
    gesture_module = ExpBaselineModule(config, weight=data_module.train_dataset.weight)
    logger = TensorBoardLogger("tb_logs", name=config.name)
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        callbacks=[ ModelCheckpoint(every_n_epochs=20, save_top_k=1), ],
        default_root_dir='./log',
        logger=logger,
    )
    tuner = Tuner(trainer)
    trainer.fit(model=gesture_module, datamodule=data_module)

if __name__ == '__main__':
    Fire(train)