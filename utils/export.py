import os
import torch

def export(exp, save_root):
    os.makedirs(save_root, exist_ok=True)
    for run in os.listdir(exp):
        if os.path.isdir(os.path.join(exp, run)):
            folder = os.path.join(exp, run, 'checkpoints')
            try:
                for c in sorted(os.listdir(folder), key = lambda x: int(x[:-5].split('=')[-1])):
                    if c.endswith('ckpt'):
                        ckpt = torch.load(os.path.join(folder, c))
                        new_dict = { }
                        for k, v in ckpt['state_dict'].items():
                            if k.startswith('model'):
                                new_dict[k.replace('model.', '')] = v

                        torch.save(new_dict, os.path.join(save_root, run + '.pt'))
                        print('Export', os.path.join(folder, c), os.path.join(save_root, run + '.pt'))
                        break
            except Exception as e:
                print('Error', e)

if __name__ == '__main__':
    root = './tb_logs'
    save_root = './checkpoints'
    for exp in os.listdir(root):
        if os.path.isdir(os.path.join(root, exp)):
            export(os.path.join(root, exp), os.path.join(save_root, exp))
