import argparse
import torch
from pathlib import Path

"""
从ckpt文件中提取SenseLiteV2模型并保存为pth格式

Args:
    ckpt_path: checkpoint文件路径
    output_path: 输出pth文件路径，如果为None则自动生成
"""
def extract_model_from_ckpt(ckpt_path: str, output_path: str = None):
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' not in ckpt:
        raise ValueError("Checkpoint does not contain 'state_dict' key")
    
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if any(unwanted in key for unwanted in ['criterion', 'optimizer', 'scheduler']):
            continue
        
        if key.startswith('model.'):
            new_key = key[6:]
        else:
            new_key = key
        
        new_state_dict[new_key] = value
    
    if output_path is None:
        ckpt_file = Path(ckpt_path)
        output_path = ckpt_file.parent / f"{ckpt_file.stem}.pth"
    
    print(f"Saving model to: {output_path}")
    torch.save(new_state_dict, output_path)
    print(f"Successfully exported model with {len(new_state_dict)} layers")
    
    return output_path
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract SenseLiteV2 model from checkpoint and save as pth')
    
    parser.add_argument('ckpt_path', type=str, help='Path to the checkpoint file (.ckpt)')
    
    parser.add_argument('-o', '--output', type=str, default=None, help='Output path for the pth file (default: same directory as ckpt with .pth extension)')
    
    args = parser.parse_args()
    
    extract_model_from_ckpt(args.ckpt_path, args.output)

