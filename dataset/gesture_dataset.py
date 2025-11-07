import os
import numpy as np
import torch
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


class GestureDataset(Dataset):
    def __init__(self, x_files: list[str], y_files: list[str], valid: list[int], do_aug: bool = False) -> None:
        self.do_aug = do_aug
        self.xs = None
        self.ys = None
        self.augmentor = GestureAugmentor() if do_aug else None
        for x_f, y_f in zip(x_files, y_files):
            x: np.ndarray = np.load(x_f).astype(np.float32)
            y: np.ndarray = np.load(y_f).astype(np.long)
            if self.xs is None:
                self.xs = x
            else:
                self.xs = np.concatenate([self.xs, x], axis=0)
            if self.ys is None:
                self.ys = y
            else:
                self.ys = np.concatenate([self.ys, y], axis=0)

        # retain only valid categories
        self.weight = [0 for _ in range(len(valid))]
        after_xs = []
        after_ys = []
        for i in range(len(self.ys)):
            for vi, v in enumerate(valid):
                if self.ys[i] == v:
                    self.ys[i] = vi
                    after_xs.append(self.xs[i])
                    after_ys.append(self.ys[i])
                    self.weight[int(self.ys[i])] += 1
                    break
        self.xs = np.array(after_xs)
        self.ys = np.array(after_ys)

        # gesture_labels = [
        #     'none', 'wave_right', 'wave_down', 'wave_left', 'wave_up', 'tap_air', 'tap_plane', 'push_forward',
        #     'pinch', 'clench', 'flip', 'wrist_clockwise', 'wrist_counterclockwise', 'circle_clockwise',
        #     'circle_counterclockwise', 'clap', 'snap', 'thumb_up', 'middle_pinch', 'index_flick', 'touch_plane',
        #     'thumb_tap_index', 'index_bend_and_straighten', 'ring_pinch', 'pinky_pinch', 'slide_plane',
        #     'pinch_down', 'pinch_up', 'boom', 'tap_up', 'throw', 'touch_left', 'touch_right', 'slide_up',
        #     'slide_down', 'slide_left', 'slide_right', 'aid_slide_left', 'aid_slide_right', 'touch_up',
        #     'touch_down', 'touch_ring', 'long_touch_ring', 'spread_ring'
        # ]

        # print("weight:", self.weight)
        # for i in range(len(self.weight)):
        #     print(gesture_labels[i], self.weight[i])

        # exit(0)

        # get weights for balancing
        w_min = 100000000
        for i in range(len(self.weight)):
            if self.weight[i] > 0 and self.weight[i] < w_min:
                w_min = self.weight[i]
        for i in range(len(self.weight)):
            if self.weight[i] > 0:
                self.weight[i] = w_min / self.weight[i]
        self.weight = torch.FloatTensor(self.weight)
        self.length = self.xs.shape[0]

        print('Weight:', self.weight)
        print('The shape of xs:', self.xs.shape)
        print('Length of the dataset:', self.length)

    def augment(self, x: np.ndarray) -> np.ndarray:
        if self.augmentor is None:
            return x
        return self.augmentor(x)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if self.do_aug:
            return self.augment(self.xs[index]), self.ys[index]
        return self.xs[index], self.ys[index]