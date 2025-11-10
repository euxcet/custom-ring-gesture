import sys
from pathlib import Path

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from utils.config import ExpBaselineTrainConfig
from dataset.exp_baseline_dataset import ExpBaselineDataset
from utils.train_utils import get_labels_id

def visual(x, filename: str):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x[i], label=f'Line {i+1}')
    plt.title('Line Plots for Rows 1-3')
    plt.xlabel('Index (Columns)')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(3, 6):
        plt.plot(x[i], label=f'Line {i+4}')
    plt.title('Line Plots for Rows 4-6')
    plt.xlabel('Index (Columns)')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    config = ExpBaselineTrainConfig.from_yaml('config/experiment/baseline.yaml')

    custom_labels_id = get_labels_id(config.labels, config.custom_labels)
    dataset = ExpBaselineDataset(dataset_type='valid', x_files=config.valid_x_files, y_files=config.valid_y_files, custom_labels_id=custom_labels_id, do_aug=config.do_aug)

    for i, (x, y) in enumerate(dataset):
        if y == 1:
            # print(y)
            visual(x, f'visual_result/visual_{i}.png')
            for r in range(5):
                visual(dataset.augment(x), f'visual_result/visual_{i}_augment_{r}.png')
            break
