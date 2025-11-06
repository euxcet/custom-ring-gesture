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

from model import get_model
from dataset.exp_directly_train_dataset import ExpDirectlyTrainDataset
from utils.config import ExpDirectlyTrainConfig

torch.set_float32_matmul_precision("medium")

def get_labels_id(labels: list[str], use_labels: list[str]) -> list[int]:
    return [labels.index(label) for label in use_labels]

class ExpBaselineDataModule(LightningDataModule):
    def __init__(self, config: ExpDirectlyTrainConfig):
        super().__init__()
        train_labels_id = get_labels_id(config.labels, config.train_labels)
        custom_labels_id = get_labels_id(config.labels, config.custom_labels)
        self.config = config
        self.train_dataset = ExpDirectlyTrainDataset(
            dataset_type='train',
            x_files=config.train_x_files,
            y_files=config.train_y_files,
            train_labels_id=train_labels_id,
            custom_labels_id=custom_labels_id,
            custom_num_samples=config.custom_num_samples,
        )
        self.valid_dataset = ExpDirectlyTrainDataset(
            dataset_type='valid',
            x_files=config.valid_x_files,
            y_files=config.valid_y_files,
            train_labels_id=train_labels_id,
            custom_labels_id=custom_labels_id,
        )
    
    def setup(self, stage):
        ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=8, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.batch_size, num_workers=8, shuffle=False, persistent_workers=True)

class ExpBaselineModule(LightningModule):
    def __init__(self, config: ExpDirectlyTrainConfig, weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.model = get_model(config)
        self.accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        if config.balance_samples:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.num_train_labels = len(config.train_labels)
        self.num_custom_labels = len(config.custom_labels)
        self.reduced_num_classes = 1 + self.num_custom_labels
        
        self.validation_predictions = []
        self.validation_targets = []
        
        self.f1_score = F1Score(task="multiclass", num_classes=self.reduced_num_classes, average="macro")
        self.accuracy_reduced = Accuracy(task="multiclass", num_classes=self.reduced_num_classes)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.reduced_num_classes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)

        self.log(f'train loss', loss, prog_bar=True)
        self.log(f'train accuracy', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        acc = self.accuracy(output, y)

        preds = torch.argmax(output, dim=1)
        self.validation_predictions.append(preds.detach().cpu())
        self.validation_targets.append(y.detach().cpu())

        self.log(f'valid loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'valid accuracy', acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_predictions.clear()
        self.validation_targets.clear()
        self.accuracy_reduced.reset()
        self.f1_score.reset()
        self.confusion_matrix.reset()
    
    def on_validation_epoch_end(self):
        if len(self.validation_predictions) == 0:
            return
        
        all_preds = torch.cat(self.validation_predictions, dim=0)
        all_targets = torch.cat(self.validation_targets, dim=0)
        
        def map_labels(labels: torch.Tensor) -> torch.Tensor:
            mapped = labels.clone()
            mask_train = labels < self.num_train_labels
            mapped[mask_train] = 0
            mask_custom = labels >= self.num_train_labels
            mapped[mask_custom] = labels[mask_custom] - self.num_train_labels + 1
            return mapped
        
        mapped_preds = map_labels(all_preds).to(self.device)
        mapped_targets = map_labels(all_targets).to(self.device)
        
        self.accuracy_reduced.update(mapped_preds, mapped_targets)
        self.f1_score.update(mapped_preds, mapped_targets)
        self.confusion_matrix.update(mapped_preds, mapped_targets)
        
        acc_reduced = self.accuracy_reduced.compute()
        f1 = self.f1_score.compute()
        cm = self.confusion_matrix.compute()
        
        self.log('valid/accuracy_reduced', acc_reduced, sync_dist=True)
        self.log('valid/f1_score', f1, sync_dist=True)
        
        if self.trainer.is_global_zero:
            print(f"\n=== Validation Statistics (Reduced Classes) ===")
            print(f"Accuracy: {acc_reduced.item():.4f}")
            print(f"F1 Score (macro): {f1.item():.4f}")
            print(f"Confusion Matrix:")
            print(cm.cpu().numpy())
            print(f"=============================================\n")

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr, eps=self.config.eps)
        return optimizer

def setdefault(config, attr, value) -> None:
    if not hasattr(config, attr):
        setattr(config, attr, value)

def train(config: str):
    config: ExpDirectlyTrainConfig = ExpDirectlyTrainConfig.from_yaml(config)
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