import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

from dataset.gesture_augmentor import GestureAugmentor

# 0 none 1 wave_right 2 wave_down 3 wave_left 4 wave_up 5 tap_air
# 6 tap_plane 7 push_forward 8 pinch 9 clench 10 flip 11 wrist_clockwise
# 12 wrist_counterclockwise 13 circle_clockwise 14 circle_counterclockwise 15 clap 16 snap
# 17 thumb_up 18 middle_pinch 19 index_flick 20 touch_plane 21 thumb_tap_index
# 22 index_bend_and_straighten 23 ring_pinch 24 pinky_pinch 25 slide_plane 26 pinch_down
# 27 pinch_up 28 boom 29 tap_up 30 throw 31 touch_left 32 touch_right 33 slide_up
# 34 slide_down 35 slide_left 36 slide_right 37 aid_slide_left 38 aid_slide_right 39 touch_up
# 40 touch_down 41 touch_ring 42 long_touch_ring 43 spread_ring


class PairDataset(Dataset):
    def __init__(
        self,
        x_files: list[str],
        y_files: list[str],
        use_labels_id: list[int],
        do_aug: bool = False,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ) -> None:
        self.do_aug = do_aug
        self.augmentor = GestureAugmentor() if do_aug else None
        self.num_classes = len(use_labels_id)

        self.xs, self.ys = self.load_from_files(x_files, y_files)
        self.xs, self.ys = self.filter_data(self.xs, self.ys, use_labels_id, keep_index=False)

        self.groups = self.get_groups(self.xs, self.ys)
        self.weight = self.get_weight(self.ys)

        self.mean = mean
        self.std = std

        self.length = self.xs.shape[0]
        print('Weight:', self.weight)
        print('The shape of xs:', self.xs.shape)
        print('Length of the dataset:', self.length)

    def get_groups(self, xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
        groups = [[] for i in range(self.num_classes)]
        for i in range(len(ys)):
            groups[ys[i]].append(xs[i])
        return groups

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

    def get_weight(self, ys: np.ndarray) -> torch.Tensor:
        counts = np.bincount(ys.astype(int), minlength=self.num_classes)
        print("counts:", counts)
        valid_counts = counts[counts > 0]
        w_max = valid_counts.max() if len(valid_counts) > 0 else 1
        weights = np.divide(w_max, counts, out=np.zeros_like(counts, dtype=float), where=counts>0)
        return torch.FloatTensor(weights)

    def augment(self, x: np.ndarray) -> np.ndarray:
        if self.augmentor is None:
            return x
        return self.augmentor(x)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        group_index = random.randint(0, len(self.groups) - 1)
        group = self.groups[group_index]
        x_i, x_j = random.sample(group, 2)
        x_i = (x_i - self.mean[:, None]) / self.std[:, None]
        x_j = (x_j - self.mean[:, None]) / self.std[:, None]
        return x_i, x_j