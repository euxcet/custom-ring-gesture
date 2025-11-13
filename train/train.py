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

from dataset.gesture_dataset import GestureDataset
from model import get_model
from utils.config import TrainConfig
from utils.train_utils import get_labels_id

torch.set_float32_matmul_precision("medium")

class GestureDataModule(LightningDataModule):
    def __init__(self, config: TrainConfig):
        super().__init__()
        use_labels_id = get_labels_id(config.labels, config.use_labels)
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
    def __init__(self, config: TrainConfig, weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.model = get_model(config)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.valid_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.train_micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="micro")
        self.valid_micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="micro")
        self.train_macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=config.num_classes)
        if config.balance_samples:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.train_accuracy(output, y)
        self.train_micro_f1(output, y)
        self.train_macro_f1(output, y)
        self.log(f'train loss', loss, prog_bar=True)
        self.log(f'train accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train micro f1', self.train_micro_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train macro f1', self.train_macro_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.train_accuracy.reset()
        self.train_micro_f1.reset()
        self.train_macro_f1.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.valid_accuracy(output, y)
        self.valid_micro_f1(output, y)
        self.valid_macro_f1(output, y)
        preds = torch.argmax(output, dim=1)
        self.confusion_matrix.update(preds, y)

        self.log(f'valid loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'valid accuracy', self.valid_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'valid micro f1', self.valid_micro_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'valid macro f1', self.valid_macro_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        confusion = self.confusion_matrix.compute()
        confusion_cpu = confusion.detach().cpu()
        confusion_np = confusion_cpu.numpy()
        with np.printoptions(threshold=np.inf, linewidth=1_000_000):
            matrix_str = np.array2string(confusion_np, separator=" ")
        self.print(f"\nValidation confusion matrix:\n{matrix_str}")
        self.confusion_matrix.reset()
        self.valid_accuracy.reset()
        self.valid_micro_f1.reset()
        self.valid_macro_f1.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr, eps=self.config.eps)
        return optimizer

def setdefault(config, attr, value) -> None:
    if not hasattr(config, attr):
        setattr(config, attr, value)

def train(config: str):
    config: TrainConfig = TrainConfig.from_yaml(config)
    data_module = GestureDataModule(config)
    gesture_module = GestureModule(config, weight=data_module.train_dataset.weight)
    logger = TensorBoardLogger("tb_logs", name=config.name)
    print("Number of classes:", config.num_classes)
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.epoch,
        callbacks=[ ModelCheckpoint(every_n_epochs=20, save_top_k=1), ],
        default_root_dir='./log',
        logger=logger,
    )
    tuner = Tuner(trainer)
    trainer.fit(model=gesture_module, datamodule=data_module)
