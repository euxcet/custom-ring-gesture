import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.model import load_model_from_checkpoint
from utils.config import TrainConfig

def evaluate_sample(config, model, sample: np.ndarray):
    channel_size = 6
    window_size = 200
    confidence_threshold = 0.9
    window = []
    print(sample.shape)
    for i in range(sample.shape[0]):
        window.append(sample[i][:channel_size])
        while len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            input_tensor = torch.tensor(np.array(window).T.reshape(1, channel_size, window_size))
            output_tensor = F.softmax(model(input_tensor).detach().cpu(), dim=1)
            gesture_id = torch.max(output_tensor, dim=1)[1].item()
            confidence = output_tensor[0][gesture_id].item()
            if confidence > confidence_threshold:
                print(gesture_id, config.labels[gesture_id])

def evaluate(config: str, checkpoint: str = './checkpoints/gesture/'):
    model = load_model_from_checkpoint(TrainConfig.from_yaml(config), torch.load(checkpoint))