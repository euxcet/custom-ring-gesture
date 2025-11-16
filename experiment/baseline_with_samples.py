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

from model import get_model, use_pretrained_model, freeze_model, load_model_from_checkpoint
from dataset.exp_baseline_with_samples_dataset import ExpBaselineWithSamplesDataset
from utils.config import ExpBaselineTrainConfig
from utils.train_utils import get_labels_id

# torch.set_float32_matmul_precision("medium")

class ExpBaselineDataModule(LightningDataModule):
    def __init__(self, config: ExpBaselineTrainConfig, samples: list[list[torch.Tensor]]):
        super().__init__()
        self.config = config
        self.train_dataset = ExpBaselineWithSamplesDataset(
            dataset_type='train',
            x_files=config.train_x_files,
            y_files=config.train_y_files,
            samples=samples,
            do_aug=config.do_aug,
            do_vae_aug=config.do_vae_aug,
            vae_model_path=config.vae_model_path,
            vae_latent_dim=config.vae_latent_dim,
            do_repeat=config.do_repeat,
        )
    
    def setup(self, stage):
        ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=8, shuffle=True, persistent_workers=True)


class ExpBaselineModule(LightningModule):
    def __init__(self, config: ExpBaselineTrainConfig, weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.model = get_model(config)
        if config.use_pretrained_model:
            self.model = use_pretrained_model(self.model, config)
        if config.do_freeze_model:
            self.model = freeze_model(self.model)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.train_micro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="micro")
        self.train_macro_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.train_confmat = ConfusionMatrix(task="multiclass", num_classes=config.num_classes)

        if config.balance_samples:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.num_custom_labels = len(config.custom_labels)
        
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.train_accuracy(output, y)
        self.train_micro_f1(output, y)
        self.train_macro_f1(output, y)
        self.train_confmat.update(output, y)

        self.log(f'train loss', loss, prog_bar=True)
        self.log(f'train accuracy', self.train_accuracy, prog_bar=True, sync_dist=True)
        self.log(f'train micro f1', self.train_micro_f1, prog_bar=False, sync_dist=True)
        self.log(f'train macro f1', self.train_macro_f1, prog_bar=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        try:
            train_cm = self.train_confmat.compute()
            print("\n========== Train Confusion Matrix ==========")
            print(train_cm.cpu().numpy())
            print("===========================================\n")
        except Exception as e:
            print(f"Failed to compute train confusion matrix: {e}")

        self.train_accuracy.reset()
        self.train_micro_f1.reset()
        self.train_macro_f1.reset()
        self.train_confmat.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr, eps=self.config.eps)
        return optimizer

def setdefault(config, attr, value) -> None:
    if not hasattr(config, attr):
        setattr(config, attr, value)

def train(config: str, samples: list[list[torch.Tensor]]):
    config: ExpBaselineTrainConfig = ExpBaselineTrainConfig.from_yaml(config)
    config.num_classes = len(samples) + 1
    data_module = ExpBaselineDataModule(config, samples)
    gesture_module = ExpBaselineModule(config, weight=data_module.train_dataset.weight)
    logger = TensorBoardLogger("tb_logs", name=config.name)
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
    
    # save final trained model
    export_dir = project_root / "export_models"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "model.cpkt"
    trainer.save_checkpoint(str(export_path), weights_only=False)
    print(f"Final checkpoint saved to {export_path}")
    
    if config.use_pretrained_model:
        config.pretrained_checkpoint_path = export_path
        model = load_model_from_checkpoint(config, torch.load('export_models/model.cpkt'))
        # save model
        export_model_path = export_dir / "model.pth"
        torch.save(model.state_dict(), export_model_path)
        print(f"Final model saved to {export_model_path}")

if __name__ == '__main__':
    Fire(train)
