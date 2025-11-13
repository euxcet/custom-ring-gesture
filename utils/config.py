from __future__ import annotations
from omegaconf import OmegaConf

class ModelConfig:
    def __init__(
        self,
        name: str,
        input_channels: int,
        depth: int,
    ):
        self.name = name
        self.input_channels = input_channels
        self.depth = depth

class TrainConfig:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        epoch: int,
        lr: float,
        eps: float,
        device: str,
        batch_size: int,
        balance_samples: bool,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        labels: list[str],
        use_labels: list[str],
    ) -> None:
        self.name = name
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.eps = eps
        self.device = device
        self.batch_size = batch_size
        self.balance_samples = balance_samples
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.use_labels = use_labels
        self.num_classes = len(use_labels)

    @staticmethod
    def from_yaml(path: str) -> TrainConfig:
        config = OmegaConf.load(path)
        return TrainConfig(**config)

class ExpDirectlyTrainConfig:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        lr: float,
        eps: float,
        device: str,
        batch_size: int,
        balance_samples: bool,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        labels: list[str],
        train_labels: list[str],
        custom_labels: list[str],
        custom_num_samples: int,
    ) -> None:
        self.name = name
        self.model = model
        self.lr = lr
        self.eps = eps
        self.device = device
        self.batch_size = batch_size
        self.balance_samples = balance_samples
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.train_labels = train_labels
        self.custom_labels = custom_labels
        self.custom_num_samples = custom_num_samples
        self.num_classes = len(train_labels) + len(custom_labels)

    @staticmethod
    def from_yaml(path: str) -> ExpDirectlyTrainConfig:
        config = OmegaConf.load(path)
        return ExpDirectlyTrainConfig(**config)

class ExpBaselineTrainConfig:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        epoch: int,
        lr: float,
        eps: float,
        device: str,
        batch_size: int,
        balance_samples: bool,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        pretrained_checkpoint_path: str,
        use_pretrained_model: bool,
        labels: list[str],
        custom_labels: list[str],
        custom_num_samples: int,
        do_aug: bool,
        do_vae_aug: bool,
        vae_model_path: str,
        vae_latent_dim: int,
        do_repeat: bool,
        do_freeze_model: bool,
    ) -> None:
        self.name = name
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.eps = eps
        self.device = device
        self.batch_size = batch_size
        self.balance_samples = balance_samples
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.use_pretrained_model = use_pretrained_model
        self.custom_labels = custom_labels
        self.custom_num_samples = custom_num_samples
        self.do_aug = do_aug
        self.do_vae_aug = do_vae_aug
        self.vae_model_path = vae_model_path
        self.vae_latent_dim = vae_latent_dim
        self.num_classes = len(custom_labels)
        self.do_repeat = do_repeat
        self.do_freeze_model = do_freeze_model

    @staticmethod
    def from_yaml(path: str) -> ExpBaselineTrainConfig:
        config = OmegaConf.load(path)
        return ExpBaselineTrainConfig(**config)

class VaeTrainConfig:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        latent_dim: int,
        beta: float,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        labels: list[str],
        use_labels: list[str],
        split_ratio: float,
        seed: int,
        save_path: str,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.beta = beta
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.use_labels = use_labels
        self.split_ratio = split_ratio
        self.seed = seed
        self.save_path = save_path

    @staticmethod
    def from_yaml(path: str) -> VaeTrainConfig:
        config = OmegaConf.load(path)
        return VaeTrainConfig(**config)

class VaeTrainDeltaConfig:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        hidden: int,
        latent_dim: int,
        beta: float,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        labels: list[str],
        use_labels: list[str],
        split_ratio: float,
        seed: int,
        save_path: str,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.beta = beta
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.use_labels = use_labels
        self.split_ratio = split_ratio
        self.seed = seed
        self.save_path = save_path

    @staticmethod
    def from_yaml(path: str) -> VaeTrainDeltaConfig:
        config = OmegaConf.load(path)
        return VaeTrainDeltaConfig(**config)


class VaeGenerateConfig:
    def __init__(
        self,
        train_x_files: list[str],
        train_y_files: list[str],
        valid_x_files: list[str],
        valid_y_files: list[str],
        labels: list[str],
        use_labels: list[str],
        vae_model_path: str,
        latent_dim: int, # TODO: remove this and beta
    ) -> None:
        self.train_x_files = train_x_files
        self.train_y_files = train_y_files
        self.valid_x_files = valid_x_files
        self.valid_y_files = valid_y_files
        self.labels = labels
        self.use_labels = use_labels
        self.vae_model_path = vae_model_path
        self.latent_dim = latent_dim

    @staticmethod
    def from_yaml(path: str) -> VaeGenerateConfig:
        config = OmegaConf.load(path)
        return VaeGenerateConfig(**config)