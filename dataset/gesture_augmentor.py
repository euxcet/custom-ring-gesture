from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class GestureAugmentorConfig:
    noise_std: float = 0.1
    noise_prob: float = 1
    channel_scale_range: tuple[float, float] = (0.5, 2)
    channel_scale_prob: float = 1
    time_shift_ratio: float = 0.2
    time_shift_prob: float = 1
    temporal_mask_prob: float = 0.3
    temporal_mask_max_ratio: float = 0.1

class GestureAugmentor:
    def __init__(self, config: Optional[GestureAugmentorConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or GestureAugmentorConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[0] != 6:
            raise ValueError(f"Expected input of shape (6, T), got {x.shape}")

        augmented = np.array(x, copy=True)
        augmented = augmented.astype(np.float32, copy=False)

        if self._should_apply(self.config.noise_prob):
            augmented = self._add_noise(augmented, self.config.noise_std)

        if self._should_apply(self.config.channel_scale_prob):
            augmented = self._channel_scale(augmented, self.config.channel_scale_range)

        if self._should_apply(self.config.time_shift_prob):
            augmented = self._time_shift(augmented, self.config.time_shift_ratio)

        if self._should_apply(self.config.temporal_mask_prob):
            augmented = self._temporal_mask(augmented, self.config.temporal_mask_max_ratio)

        return augmented

    def _should_apply(self, prob: float) -> bool:
        return prob > 0.0 and self.rng.random() < prob

    def _add_noise(self, x: np.ndarray, std: float) -> np.ndarray:
        noise = self.rng.normal(0.0, std, size=x.shape).astype(np.float32)
        return x + noise

    def _channel_scale(self, x: np.ndarray, scale_range: tuple[float, float]) -> np.ndarray:
        low, high = scale_range
        scale = self.rng.uniform(low, high, size=(x.shape[0], 1)).astype(np.float32)
        return x * scale

    def _time_shift(self, x: np.ndarray, ratio: float) -> np.ndarray:
        max_shift = max(1, int(x.shape[1] * ratio))
        shift = self.rng.integers(-max_shift, max_shift + 1)
        if shift == 0:
            return x
        return np.roll(x, shift=shift, axis=1)

    def _temporal_mask(self, x: np.ndarray, max_ratio: float) -> np.ndarray:
        length = x.shape[1]
        max_width = max(1, int(length * max_ratio))
        width = self.rng.integers(1, max_width + 1)
        start = self.rng.integers(0, max(1, length - width + 1))
        mask_slice = slice(start, start + width)
        x_copy = x.copy()
        x_copy[:, mask_slice] = 0.0
        return x_copy
