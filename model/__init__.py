import torch
import torch.nn as nn
from utils.config import TrainConfig, ExpBaselineTrainConfig
from copy import deepcopy

from .model import InceptionTimeModel, InceptionTimeMoreFcModel
from .sense_lite_v2 import SenseLiteV2

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
    elif config.model.name == 'SenseLiteV2':
        return SenseLiteV2(
            input_len=200,
            input_dim=config.model.input_channels,
            output_dim=config.num_classes,
            conv_dim=[96, 96, 96],
            dropout=0.1,
            fc_dim=256,
            sca_reduction=8,
            n_input_filters=32,
        )

def load_state_dict_from_checkpoint(checkpoint: dict) -> dict:
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key, value in state_dict.items():
        if not any(unwanted in key for unwanted in ['criterion', 'optimizer', 'scheduler']):
            if key.startswith('model.'):
                new_key = key[6:]
            else:
                new_key = key
            new_state_dict[new_key] = value

    return new_state_dict

def load_model_from_checkpoint(config: TrainConfig, checkpoint: dict) -> nn.Module:
    model = get_model(config)
    model.load_state_dict(load_state_dict_from_checkpoint(checkpoint))
    # state_dict = checkpoint['state_dict']
    # new_state_dict = {}

    # for key, value in state_dict.items():
    #     if not any(unwanted in key for unwanted in ['criterion', 'optimizer', 'scheduler']):
    #         if key.startswith('model.'):
    #             new_key = key[6:]
    #         else:
    #             new_key = key
    #         new_state_dict[new_key] = value

    # model.load_state_dict(new_state_dict)

    return model

def use_pretrained_model(model: nn.Module, config: ExpBaselineTrainConfig) -> nn.Module:
    pretrained_config = deepcopy(config)
    pretrained_config.num_classes = 27
    # pretrained_model = load_model_from_checkpoint(pretrained_config, torch.load(config.pretrained_checkpoint_path))
    # pretrained_state_dict = pretrained_model.state_dict()
    pretrained_state_dict = load_state_dict_from_checkpoint(torch.load(config.pretrained_checkpoint_path))
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
    
    return model

def freeze_model(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model
    