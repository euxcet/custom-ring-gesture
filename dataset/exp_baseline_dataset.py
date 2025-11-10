import numpy as np
import torch
import random
from torch.utils.data import Dataset
from dataset.gesture_augmentor import GestureAugmentor
from model.vae import VAE

class ExpBaselineDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        x_files: list[str],
        y_files: list[str],
        custom_labels_id: list[int],
        custom_num_samples: list[int] = None,
        do_aug: bool = False,
        do_vae_aug: bool = False,
        vae_model_path: str = None,
        vae_latent_dim: int = 128,
        do_repeat: bool = False,
    ) -> None:
        self.device = torch.device("cuda")
        self.num_classes = len(custom_labels_id)
        self.do_aug = do_aug
        self.do_vae_aug = do_vae_aug
        self.do_repeat = do_repeat
        self.augmentor = GestureAugmentor() if do_aug else None
        
        if do_vae_aug:
            self.load_vae_model(vae_model_path, vae_latent_dim)

        self.xs, self.ys = self.load_from_files(x_files, y_files)
        if dataset_type == 'train':
            self.xs, self.ys = self.filter_data(self.xs, self.ys, custom_labels_id, keep_index=True)
            self.xs, self.ys = self.keep_custom_data(self.xs, self.ys, custom_labels_id, num_samples=custom_num_samples)
            self.xs, self.ys = self.filter_data(self.xs, self.ys, custom_labels_id, keep_index=False)
        else:
            self.xs, self.ys = self.filter_data(self.xs, self.ys, custom_labels_id, keep_index=False)

        # if self.do_aug:
        #     self.xs, self.ys = self.augment_data(self.xs, self.ys, 7000)
        if self.do_vae_aug:
            self.xs, self.ys = self.vae_augment_data(self.xs, self.ys, 7000)
        if self.do_repeat:
            self.xs, self.ys = self.repeat_data(self.xs, self.ys, 7000)

        self.weight = self.get_weight(self.ys)
        self.length = self.xs.shape[0]

        print('Weight:', self.weight)
        print('The shape of xs:', self.xs.shape)
        print('Length of the dataset:', self.length)

    def load_vae_model(self, vae_model_path: str, vae_latent_dim: int):
        ckpt = torch.load(vae_model_path, map_location=self.device, weights_only=False)
        self.vae = VAE(latent_dim=vae_latent_dim, beta=ckpt['args'].get('beta', 1.0)).to(self.device)
        self.vae.load_state_dict(ckpt['model_state'])
        self.vae.eval()

        self.vae_mean = ckpt['mean']
        self.vae_std = ckpt['std']

    def load_from_files(self, x_files: list[str], y_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
        # merge all x and y
        xs, ys = None, None
        for x_f, y_f in zip(x_files, y_files):
            x: np.ndarray = np.load(x_f).astype(np.float32)
            y: np.ndarray = np.load(y_f).astype(np.long)
            xs = np.concatenate([xs, x], axis=0) if xs is not None else x
            ys = np.concatenate([ys, y], axis=0) if ys is not None else y
        return xs, ys

    def filter_data(self, x: np.ndarray, y: np.ndarray, filter_labels: list[int], keep_index: bool = False) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(len(y)):
            for vi, v in enumerate(filter_labels):
                if y[i] == v:
                    xs.append(x[i])
                    ys.append(y[i] if keep_index else vi)
                    break
        return np.array(xs), np.array(ys)

    def keep_custom_data(self, x: np.ndarray, y: np.ndarray, labels_id: list[int], num_samples: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        remain = {v: num_samples[vi] for vi, v in enumerate(labels_id)}
        for i in range(len(y)):
            if y[i] in labels_id:
                if remain[y[i]] > 0:
                    xs.append(x[i])
                    ys.append(y[i])
                    remain[y[i]] -= 1
                else:
                    continue
            else:
                xs.append(x[i])
                ys.append(y[i])
        return np.array(xs), np.array(ys)

    def get_weight(self, ys: np.ndarray) -> torch.Tensor:
        counts = np.bincount(ys.astype(int), minlength=self.num_classes)
        print("counts:", counts)
        valid_counts = counts[counts > 0]
        w_max = valid_counts.max() if len(valid_counts) > 0 else 1
        weights = np.divide(w_max, counts, out=np.zeros_like(counts, dtype=float), where=counts>0)
        return torch.FloatTensor(weights)

    def repeat_data(self, xs: np.ndarray, ys: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.do_repeat:
            return xs, ys

        augmented_xs = list()
        augmented_ys = list()
        for i in range(len(ys)):
            if ys[i] != 0:
                for j in range(num_samples):
                    augmented_xs.append(xs[i])
                    augmented_ys.append(ys[i])
            else:
                augmented_xs.append(xs[i])
                augmented_ys.append(ys[i])

        return np.array(augmented_xs), np.array(augmented_ys)

    def augment(self, x: np.ndarray) -> np.ndarray:
        if self.augmentor is None:
            return x
        return self.augmentor(x)

    def augment_data(self, xs: np.ndarray, ys: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.do_aug or self.augmentor is None:
            return xs, ys

        augmented_xs = list()
        augmented_ys = list()
        for i in range(len(ys)):
            if ys[i] != 0:
                for j in range(num_samples):
                    augmented_sample = self.augment(xs[i])
                    augmented_xs.append(augmented_sample)
                    augmented_ys.append(ys[i])
            else:
                augmented_xs.append(xs[i])
                augmented_ys.append(ys[i])

        return np.array(augmented_xs), np.array(augmented_ys)

    def vae_augment(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        x0, _, _ = self.vae(x)
        x0 = x0.detach().cpu().numpy().squeeze(0)
        x0 = x0 * self.vae_std[:, None] + self.vae_mean[:, None]
        return x0


    def vae_augment_data(self, xs: np.ndarray, ys: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.do_vae_aug or self.vae is None:
            return xs, ys

        augmented_xs = list()
        augmented_ys = list()
        for i in range(len(ys)):
            if ys[i] != 0:
                for j in range(num_samples):
                    augmented_sample = self.vae_augment(xs[i])
                    augmented_xs.append(augmented_sample)
                    augmented_ys.append(ys[i])
            else:
                augmented_xs.append(xs[i])
                augmented_ys.append(ys[i])

        return np.array(augmented_xs), np.array(augmented_ys)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.xs[index]
        if self.do_aug:
            x = self.augment(x)
        # if self.do_vae_aug:
        #     x = self.vae_augment(x)
        return x, self.ys[index]