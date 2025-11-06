import torch
import torch.nn as nn
from utils.config import TrainConfig, ExpBaselineTrainConfig
from copy import deepcopy

from .model import InceptionTimeModel, InceptionTimeMoreFcModel

def get_model(config) -> nn.Module:
    if config.model.name == 'InceptionTimeModel':
        return InceptionTimeModel(
            input_channels=config.model.input_channels,
            num_classes=config.num_classes,
            depth=config.model.depth,
        )
    elif config.model.name == 'InceptionTimeMoreFcModel':
        return InceptionTimeMoreFcModel(
            input_channels=config.model.input_channels,
            num_classes=config.num_classes,
            depth=config.model.depth,
        )

def load_model_from_checkpoint(config: TrainConfig, checkpoint: dict) -> nn.Module:
    model = get_model(config)

    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key, value in state_dict.items():
        if not any(unwanted in key for unwanted in ['criterion', 'optimizer', 'scheduler']):
            if key.startswith('model.'):
                new_key = key[6:]
            else:
                new_key = key
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    return model

def use_pretrained_model(model: nn.Module, config: ExpBaselineTrainConfig) -> nn.Module:
    pretrained_config = deepcopy(config)
    pretrained_config.num_classes = 44 - 6
    pretrained_model = load_model_from_checkpoint(pretrained_config, torch.load(config.pretrained_checkpoint_path))
    
    pretrained_state_dict = pretrained_model.state_dict()
    model_state_dict = model.state_dict()
  
    filtered_state_dict = {}
    for key, value in pretrained_state_dict.items():
        if not key.startswith('fc.'):
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Shape mismatch for {key}. Pretrained: {value.shape}, Model: {model_state_dict[key].shape}")
            else:
                print(f"Warning: Key {key} not found in model")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model