from __future__ import annotations
import os
import json
import math
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from utils.config import VaeTrainConfig, VaeTrainDeltaConfig, VaeGenerateConfig
from dataset.gesture_dataset import GestureDataset
from dataset.pair_dataset import PairDataset
from model.vae import VAE
from utils.train_utils import get_labels_id
from visual.vis import visual
from sklearn.cluster import KMeans

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_vae(config: VaeTrainConfig):
    device = torch.device("cuda")
    set_seed(config.seed)

    use_labels_id = get_labels_id(config.labels, config.use_labels)
    train_ds = GestureDataset(config.train_x_files, config.train_y_files, use_labels_id, do_normalize=True)
    val_ds = GestureDataset(config.valid_x_files, config.valid_y_files, use_labels_id, do_normalize=True)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8)

    model = VAE(latent_dim=config.latent_dim, beta=config.beta).to(device)
    opt = Adam(model.parameters(), lr=config.lr)

    best_val = float('inf')
    ckpt_path = os.path.join(config.save_path, f"vae_1dcnn_lat{config.latent_dim}_beta{config.beta}.pt")

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, _ in train_loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, recon, kld = VAE.loss_fn(x, x_recon, mu, logvar, beta=config.beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        # validation
        model.eval()
        with torch.no_grad():
            v_losses = []
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, mu, logvar = model(x)
                loss, _, _ = VAE.loss_fn(x, x_recon, mu, logvar, beta=config.beta)
                v_losses.append(loss.item())
            val_loss = float(np.mean(v_losses))

        print(f"[VAE] Epoch {epoch}/{config.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'mean': train_ds.mean,
                'std': train_ds.std,
                'args': vars(config),
            }, ckpt_path)
    print(f"Saved best VAE to {ckpt_path}")


def generate(config: VaeGenerateConfig):
    device = torch.device("cuda")

    ckpt = torch.load(config.vae_model_path, map_location=device, weights_only=False)
    vae = VAE(latent_dim=config.latent_dim, beta=ckpt['args'].get('beta', 1.0)).to(device)
    vae.load_state_dict(ckpt['model_state'])
    vae.eval()

    mean = ckpt['mean']
    std = ckpt['std']
    print(mean, std)

    use_labels_id = get_labels_id(config.labels, config.use_labels)
    dataset = GestureDataset(config.valid_x_files, config.valid_y_files, use_labels_id, do_normalize=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)

    for x, y in loader:
        if y == 5:
            visual(x.numpy().squeeze(0), 'visual_result/vae_original_beta0_0.png')
            x = x.to(device)
            for i in range(1):
                x0, _, _ = vae(x)
                x0 = x0.detach().cpu().numpy().squeeze(0)
                x0 = x0 * std[:, None] + mean[:, None]
                visual(x0, f'visual_result/vae_generated_beta0_{i}.png')
            # print(y)
            break
    exit(0)


def build_parser():
    p = argparse.ArgumentParser(description="VAE + Delta for 6x200 IMU")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vae = sub.add_parser("train-vae")
    p_vae.add_argument("--config", type=str, required=True)

    p_delta = sub.add_parser("train-delta")
    p_delta.add_argument("--config", type=str, required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--config", type=str, required=True)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train-vae":
        train_vae(VaeTrainConfig.from_yaml(args.config))
    elif args.cmd == "generate":
        generate(VaeGenerateConfig.from_yaml(args.config))
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
