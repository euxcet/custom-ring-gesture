#!/usr/bin/env python
"""
VAE + Delta for 6-axis IMU time series (shape: 6 x 200)
- Encoder/Decoder: 1D CNN
- Latent dim: 32
- Supports supervised (labels) or unsupervised (pseudo labels via KMeans) delta training
- End-to-end CLI:
    train-vae        Train VAE on IMU data directory
    train-delta      Train DeltaNet on same-class pairs (uses labels.csv or pseudo labels)
    generate         Given one IMU sample, generate N same-class sequences

Data expectations
-----------------
- Directory with files, each sample is (6,200)
  Supported formats per file: .npy (numpy array), .csv (200 rows x 6 cols or 6 rows x 200 cols)
- Optional labels CSV: labels.csv with columns: filename,label
- Normalization (mean/std per channel) is computed on the train set and saved to norm.json

Examples
--------
python imu_vae_delta.py train-vae --data_dir ./data --epochs 50
python imu_vae_delta.py train-delta --data_dir ./data --epochs 30 --use_pseudo_labels --num_clusters 10
python imu_vae_delta.py generate --data_dir ./data --input ./data/sample_001.npy --num_samples 8 --out_dir ./gen

"""
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

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_imu_file(path: Path) -> np.ndarray:
    """Load a single IMU file and return float32 array of shape (6,200).
    Accepts .npy or .csv (either 200x6 or 6x200)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (200, 6):
        arr = arr.T  # -> (6,200)
    if arr.shape != (6, 200):
        raise ValueError(f"File {path} has shape {arr.shape}, expected (6,200) or (200,6)")
    return arr


def load_labels_csv(data_dir: Path) -> Optional[Dict[str, str]]:
    csv_path = data_dir / "labels.csv"
    if not csv_path.exists():
        return None
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if len(header) < 2 or header[0].lower() != "filename" or header[1].lower() != "label":
            raise ValueError("labels.csv must have header: filename,label")
        for line in f:
            if not line.strip():
                continue
            filename, label = line.strip().split(",", 1)
            mapping[filename] = label
    return mapping


# ---------------------------
# Dataset
# ---------------------------
class IMUDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", split_ratio: float = 0.9,
                 normalize: bool = True, mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 file_exts: Tuple[str, ...] = (".npy", ".csv")):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize

        files = [p for p in self.data_dir.glob("**/*") if p.suffix.lower() in file_exts]
        files = sorted(files)
        if len(files) == 0:
            raise RuntimeError(f"No data files found in {data_dir} with extensions {file_exts}")

        # Deterministic split
        set_seed(42)
        idx = list(range(len(files)))
        random.shuffle(idx)
        cut = int(len(files) * split_ratio)
        if split == "train":
            self.files = [files[i] for i in idx[:cut]]
        else:
            self.files = [files[i] for i in idx[cut:]]

        # Preload to compute mean/std if needed
        if normalize and mean_std is None:
            arrs = [load_imu_file(p) for p in self.files]
            stacked = np.stack(arrs, axis=0)  # (N,6,200)
            mean = stacked.mean(axis=(0, 2))  # (6,)
            std = stacked.std(axis=(0, 2)) + 1e-8
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)
        elif normalize and mean_std is not None:
            self.mean, self.std = mean_std
            self.mean = self.mean.astype(np.float32)
            self.std = self.std.astype(np.float32)
        else:
            self.mean = np.zeros((6,), dtype=np.float32)
            self.std = np.ones((6,), dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = load_imu_file(self.files[idx])  # (6,200)
        if self.normalize:
            arr = (arr - self.mean[:, None]) / self.std[:, None]
        # Return torch tensor (6,200)
        x = torch.from_numpy(arr)
        return x, self.files[idx].name


# ---------------------------
# VAE (1D CNN, latent=32)
# ---------------------------
class Encoder1DCNN(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, stride=2, padding=2),  # 200 -> 100
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # 100 -> 50
            nn.BatchNorm1d(128), nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # 50 -> 25
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.SiLU(),
        )
        self.len_after = 25
        self.fc_mu = nn.Linear(256 * self.len_after, latent_dim)
        self.fc_logvar = nn.Linear(256 * self.len_after, latent_dim)

    def forward(self, x):
        # x: (B,6,200)
        h = self.conv(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder1DCNN(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.len_after = 25
        self.fc = nn.Linear(latent_dim, 256 * self.len_after)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),  # 25->50
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 50->100
            nn.BatchNorm1d(128), nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 100->200
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 6, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 256, self.len_after)
        x_recon = self.deconv(h)
        return x_recon


class VAE(nn.Module):
    def __init__(self, latent_dim=32, beta=1.0):
        super().__init__()
        self.encoder = Encoder1DCNN(latent_dim)
        self.decoder = Decoder1DCNN(latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_fn(x, x_recon, mu, logvar, beta=1.0):
        recon = F.mse_loss(x_recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kld, recon.item(), kld.item()


# ---------------------------
# DeltaNet (latent-space delta prediction)
# ---------------------------
class DeltaNet(nn.Module):
    def __init__(self, latent_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_i, z_j):
        # Predict delta such that z_i ≈ z_j + delta
        return self.net(torch.cat([z_i, z_j], dim=-1))


# ---------------------------
# Pair dataset for Delta training
# ---------------------------
class PairDataset(Dataset):
    def __init__(self, data_dir: Path, labels: Dict[str, str], mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        # group by label
        groups: Dict[str, List[Path]] = {}
        for fname, lab in labels.items():
            p = self.data_dir / fname
            if not p.exists():
                # try nested search
                candidates = list(self.data_dir.rglob(fname))
                if len(candidates) == 0:
                    continue
                p = candidates[0]
            groups.setdefault(lab, []).append(p)
        # build list of (i,j) pairs from each group (sampling on the fly instead of precomputing)
        self.groups = {k: v for k, v in groups.items() if len(v) >= 2}
        self.labels_list = list(self.groups.keys())
        if len(self.groups) == 0:
            raise RuntimeError("No labels with at least 2 samples to form pairs.")

    def __len__(self):
        # arbitrary large epoch length (pairs sampled on the fly)
        return sum(len(v) for v in self.groups.values())

    def __getitem__(self, idx):
        # sample a random label, then two distinct items from that label
        lab = random.choice(self.labels_list)
        files = self.groups[lab]
        p_i, p_j = random.sample(files, 2)
        x_i = load_imu_file(p_i)
        x_j = load_imu_file(p_j)
        # normalize
        x_i = (x_i - self.mean[:, None]) / self.std[:, None]
        x_j = (x_j - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(x_i), torch.from_numpy(x_j)


# ---------------------------
# Training loops
# ---------------------------

def save_norm(data_dir: Path, mean: np.ndarray, std: np.ndarray):
    with open(data_dir / "norm.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)


def load_norm(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(data_dir / "norm.json", "r") as f:
        d = json.load(f)
    return np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32)


def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    data_dir = Path(args.data_dir)

    # Datasets
    train_ds = IMUDataset(data_dir, split="train", split_ratio=args.split_ratio, normalize=True)
    val_ds = IMUDataset(data_dir, split="val", split_ratio=args.split_ratio, normalize=True,
                        mean_std=(train_ds.mean, train_ds.std))

    # Save normalization for later stages
    save_norm(data_dir, train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VAE(latent_dim=32, beta=args.beta).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    ckpt_path = data_dir / "vae_1dcnn_lat32.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, _ in train_loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, recon, kld = VAE.loss_fn(x, x_recon, mu, logvar, beta=args.beta)
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
                loss, _, _ = VAE.loss_fn(x, x_recon, mu, logvar, beta=args.beta)
                v_losses.append(loss.item())
            val_loss = float(np.mean(v_losses))

        print(f"[VAE] Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'mean': train_ds.mean,
                'std': train_ds.std,
                'args': vars(args),
            }, ckpt_path)
    print(f"Saved best VAE to {ckpt_path}")


def get_or_build_labels(args, vae: VAE, data_dir: Path, device) -> Dict[str, str]:
    labels_map = load_labels_csv(data_dir)
    if labels_map is not None and not args.use_pseudo_labels:
        print("Using provided labels.csv for delta training.")
        return labels_map

    if args.use_pseudo_labels:
        if KMeans is None:
            raise RuntimeError("scikit-learn is required for pseudo labels. Please install scikit-learn.")
        print("Building pseudo labels via KMeans on VAE latents...")
        # Encode all files to latents (mu)
        ds_all = IMUDataset(data_dir, split="train", split_ratio=1.0, normalize=True,
                            mean_std=load_norm(data_dir))
        loader = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False)
        mu_list, names = [], []
        vae.eval()
        with torch.no_grad():
            for x, name in loader:
                x = x.to(device)
                mu, logvar = vae.encoder(x)
                mu_list.append(mu.cpu().numpy())
                names.extend(name)
        feats = np.concatenate(mu_list, axis=0)
        kmeans = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=args.seed)
        clus = kmeans.fit_predict(feats)
        labels_map = {n: f"c{int(c)}" for n, c in zip(names, clus)}
        # Save pseudo labels for reuse
        with open(data_dir / "pseudo_labels.csv", "w") as f:
            f.write("filename,label\n")
            for n, lab in labels_map.items():
                f.write(f"{n},{lab}\n")
        print(f"Saved pseudo labels to {data_dir/'pseudo_labels.csv'}")
        return labels_map

    # Fallback: need labels
    raise RuntimeError("labels.csv not found. Provide labels or use --use_pseudo_labels.")


def train_delta(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    data_dir = Path(args.data_dir)

    # Load VAE
    ckpt_path = data_dir / "vae_1dcnn_lat32.pt"
    assert ckpt_path.exists(), "Train VAE first."
    ckpt = torch.load(ckpt_path, map_location=device)
    vae = VAE(latent_dim=32, beta=ckpt['args'].get('beta', 1.0)).to(device)
    vae.load_state_dict(ckpt['model_state'])
    vae.eval()

    mean, std = load_norm(data_dir)

    # Labels or pseudo-labels
    labels_map = get_or_build_labels(args, vae, data_dir, device)

    # Pair dataset
    pair_ds = PairDataset(data_dir, labels_map, mean, std)
    loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    deltanet = DeltaNet(latent_dim=32, hidden=args.hidden).to(device)
    opt = Adam(deltanet.parameters(), lr=args.lr)

    best = float('inf')
    ckpt_delta = data_dir / "deltanet_lat32.pt"

    for epoch in range(1, args.epochs + 1):
        deltanet.train()
        losses = []
        for x_i, x_j in loader:
            x_i = x_i.to(device)  # target sample
            x_j = x_j.to(device)  # source sample
            with torch.no_grad():
                mu_i, logvar_i = vae.encoder(x_i)
                mu_j, logvar_j = vae.encoder(x_j)
            delta_pred = deltanet(mu_i, mu_j)  # predict delta such that z_i ≈ z_j + delta
            z_tilde = mu_j + delta_pred
            x_hat = vae.decoder(z_tilde)
            loss = F.mse_loss(x_hat, x_i, reduction='mean')
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(deltanet.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        print(f"[Delta] Epoch {epoch}/{args.epochs}  train_recon_mse={train_loss:.6f}")
        if train_loss < best:
            best = train_loss
            torch.save({'model_state': deltanet.state_dict(), 'args': vars(args)}, ckpt_delta)
    print(f"Saved best DeltaNet to {ckpt_delta}")


# ---------------------------
# Generation: given one sample, create N same-class samples
# ---------------------------

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models & norms
    ckpt_vae = torch.load(data_dir / "vae_1dcnn_lat32.pt", map_location=device)
    vae = VAE(latent_dim=32, beta=ckpt_vae['args'].get('beta', 1.0)).to(device)
    vae.load_state_dict(ckpt_vae['model_state'])
    vae.eval()

    ckpt_delta_path = data_dir / "deltanet_lat32.pt"
    if ckpt_delta_path.exists():
        ckpt_delta = torch.load(ckpt_delta_path, map_location=device)
        deltanet = DeltaNet(latent_dim=32, hidden=ckpt_delta['args'].get('hidden', 128)).to(device)
        deltanet.load_state_dict(ckpt_delta['model_state'])
        deltanet.eval()
    else:
        deltanet = None
        print("Warning: DeltaNet not found. Will sample raw deltas from dataset if labels available.")

    mean, std = load_norm(data_dir)

    # Load input sample
    x0 = load_imu_file(Path(args.input))  # (6,200)
    x0_norm = (x0 - mean[:, None]) / std[:, None]
    x0_t = torch.from_numpy(x0_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        mu0, logvar0 = vae.encoder(x0_t)

    # Build a small memory of z_j from same class if possible
    labels_map = load_labels_csv(data_dir)
    same_class_z: List[torch.Tensor] = []
    if labels_map is not None:
        input_name = Path(args.input).name
        lab = labels_map.get(input_name, None)
        if lab is not None:
            # Collect z for same label
            # We'll scan data_dir for that label
            paths = [fn for fn, l in labels_map.items() if l == lab and fn != input_name]
            paths = paths[: min(200, len(paths))]  # cap
            for fn in paths:
                p = data_dir / fn
                if not p.exists():
                    cands = list(data_dir.rglob(fn))
                    if len(cands) == 0:
                        continue
                    p = cands[0]
                arr = load_imu_file(p)
                arr = (arr - mean[:, None]) / std[:, None]
                xt = torch.from_numpy(arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    mu, _ = vae.encoder(xt)
                    same_class_z.append(mu.squeeze(0))
    # Fallback: no labels or empty same_class_z, sample a few random z from dataset
    if len(same_class_z) == 0:
        ds_all = IMUDataset(data_dir, split="train", split_ratio=1.0, normalize=True, mean_std=(mean, std))
        loader = DataLoader(ds_all, batch_size=64, shuffle=True)
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                mu, _ = vae.encoder(x)
                for k in range(mu.size(0)):
                    same_class_z.append(mu[k].detach())
                if len(same_class_z) > 512:
                    break

    # Generate
    results = []
    for i in range(args.num_samples):
        z_j = random.choice(same_class_z).unsqueeze(0).to(device)
        if deltanet is not None:
            with torch.no_grad():
                delta = deltanet(mu0, z_j)
        else:
            with torch.no_grad():
                delta = mu0 - z_j  # simple delta
        z_new = mu0 + delta
        with torch.no_grad():
            x_gen = vae.decoder(z_new).cpu().numpy()[0]  # (6,200) normalized
        # denormalize
        x_denorm = x_gen * std[:, None] + mean[:, None]
        out_path = out_dir / f"gen_{i:03d}.npy"
        np.save(out_path, x_denorm.astype(np.float32))
        results.append(str(out_path))
    print("Generated files:\n" + "\n".join(results))


# ---------------------------
# CLI
# ---------------------------

def build_parser():
    p = argparse.ArgumentParser(description="VAE + Delta for 6x200 IMU")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vae = sub.add_parser("train-vae")
    p_vae.add_argument("--data_dir", type=str, required=True)
    p_vae.add_argument("--epochs", type=int, default=50)
    p_vae.add_argument("--batch_size", type=int, default=128)
    p_vae.add_argument("--lr", type=float, default=1e-3)
    p_vae.add_argument("--beta", type=float, default=1.0)
    p_vae.add_argument("--split_ratio", type=float, default=0.9)
    p_vae.add_argument("--cpu", action="store_true")
    p_vae.add_argument("--seed", type=int, default=42)

    p_delta = sub.add_parser("train-delta")
    p_delta.add_argument("--data_dir", type=str, required=True)
    p_delta.add_argument("--epochs", type=int, default=30)
    p_delta.add_argument("--batch_size", type=int, default=256)
    p_delta.add_argument("--lr", type=float, default=1e-3)
    p_delta.add_argument("--hidden", type=int, default=128)
    p_delta.add_argument("--use_pseudo_labels", action="store_true",
                         help="If set, build KMeans pseudo-labels on VAE latents. Requires scikit-learn.")
    p_delta.add_argument("--num_clusters", type=int, default=10)
    p_delta.add_argument("--cpu", action="store_true")
    p_delta.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--data_dir", type=str, required=True)
    p_gen.add_argument("--input", type=str, required=True)
    p_gen.add_argument("--num_samples", type=int, default=8)
    p_gen.add_argument("--out_dir", type=str, required=True)
    p_gen.add_argument("--cpu", action="store_true")
    p_gen.add_argument("--seed", type=int, default=42)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train-vae":
        train_vae(args)
    elif args.cmd == "train-delta":
        train_delta(args)
    elif args.cmd == "generate":
        generate(args)
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
