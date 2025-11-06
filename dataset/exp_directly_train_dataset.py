import numpy as np
import torch
import random
from torch.utils.data import Dataset

class ExpDirectlyTrainDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        x_files: list[str],
        y_files: list[str],
        train_labels_id: list[int],
        custom_labels_id: list[int],
        custom_num_samples: int = 5,
    ) -> None:
        self.num_classes = len(train_labels_id) + len(custom_labels_id)

        self.xs, self.ys = self.load_from_files(x_files, y_files)
        if dataset_type == 'train':
            self.xs, self.ys = self.filter_data(self.xs, self.ys, train_labels_id + custom_labels_id, keep_index=True)
            self.xs, self.ys = self.keep_custom_data(self.xs, self.ys, custom_labels_id, num_samples=custom_num_samples)
            self.xs, self.ys = self.filter_data(self.xs, self.ys, train_labels_id + custom_labels_id, keep_index=False)
        else:
            self.xs, self.ys = self.filter_data(self.xs, self.ys, train_labels_id + custom_labels_id, keep_index=False)

        self.weight = self.get_weight(self.ys)
        self.length = self.xs.shape[0]

        print('Weight:', self.weight)
        print('The shape of xs:', self.xs.shape)
        print('Length of the dataset:', self.length)

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

    def keep_custom_data(self, x: np.ndarray, y: np.ndarray, labels_id: list[int], num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        remain = {v: num_samples for v in labels_id}
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
        valid_counts = counts[counts > 0]
        # w_min = valid_counts.min() if len(valid_counts) > 0 else 1
        # weights = np.divide(w_min, counts, out=np.zeros_like(counts, dtype=float), where=counts>0)
        w_max = valid_counts.max() if len(valid_counts) > 0 else 1
        weights = np.divide(w_max, counts, out=np.zeros_like(counts, dtype=float), where=counts>0)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[index], self.ys[index]