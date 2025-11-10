# custom-ring-gesture

训练预训练模型：
```bash
uv run main.py config/experiment/pretrained_senselitev2.yaml
```

训练baseline:
```bash
uv run experiment/baseline.py --config config/experiment/baseline.yaml
```

训练vae：
```bash
uv run vae.py train-vae --config config/vae/train_vae.yaml
```