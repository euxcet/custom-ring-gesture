import os
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import get_model
from utils.file_utils import load_config
from utils.to_utils import DictToObject

def load_checkpoint(config_name: str, checkpoint_folder: str):
    candidate_file = os.path.join(checkpoint_folder, config_name + '.pt')
    print(candidate_file)
    if os.path.exists(candidate_file):
        return torch.load(candidate_file)
    raise FileNotFoundError

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

def evaluate(config: str, sample: str, checkpoint: str = './checkpoints/gesture/'):
    config_path = config
    config_name = config_path.strip().split('/')[-1].split('.')[0]
    config = DictToObject(load_config(config))
    setattr(config, 'num_classes', len(config.use_labels))
    model = get_model(config)
    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        model.load_state_dict(load_checkpoint(config_name, checkpoint))
    if os.path.isfile(sample):
        evaluate_sample(config, model, np.load(sample).astype(np.float32))

if __name__ == '__main__':
    Fire(evaluate)
