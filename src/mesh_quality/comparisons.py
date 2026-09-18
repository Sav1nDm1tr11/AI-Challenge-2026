"""Input datasets for the fixed global-image and geometry comparisons."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .data import image_views, load_backbone
from .geometry import item_sampling_seed, sample_mesh_surface


class CachedMultiViewHead(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.defect_head = nn.Linear(hidden_dim, 10)

    def forward(self, features):
        pooled = torch.cat([features.mean(1), features.amax(1)], dim=-1)
        return self.defect_head(self.fusion(pooled))


class GlobalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


@torch.inference_mode()
def global_features(root, ids, image_dir, assets, config, device):
    """Extract CLS tokens at 224; reuse matching historical features locally."""
    ids = list(ids)
    legacy = Path(root) / "checkpoints/dinov2_weighted_ap_20260917_113030/feature_cache"
    if (legacy / "train.pt").exists() and (legacy / "valid.pt").exists():
        rows = {}
        for name in ["train", "valid"]:
            cache = torch.load(
                legacy / f"{name}.pt", weights_only=True, map_location="cpu"
            )
            rows.update(zip(cache["item_ids"], cache["features"]))
        if set(ids).issubset(rows):
            return torch.stack([rows[item] for item in ids])
    folder = Path(root) / "outputs/final/global_features"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(config, resolution=224)
    signature = {
        k: cfg[k]
        for k in [
            "model_name",
            "resolution",
            "image_mean",
            "image_std",
            "source_checkpoint_sha256",
        ]
    }
    manifest = folder / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Global cache configuration changed; choose another folder.")
    manifest.write_text(json.dumps(signature))
    missing = [item for item in ids if not (folder / f"{item}.npy").exists()]
    if missing:
        encoder = load_backbone(assets, cfg, device)
        for item in tqdm(missing, desc="DINOv2 CLS 224"):
            views = image_views(Path(image_dir) / f"{item}.png", cfg).to(device)
            features = torch.cat([encoder(views[i : i + 3]) for i in range(0, 6, 3)])
            np.save(folder / f"{item}.npy", features.cpu().numpy())
        del encoder
    return torch.from_numpy(np.stack([np.load(folder / f"{item}.npy") for item in ids]))


class SurfaceDataset(Dataset):
    """Area-weighted 32768-point cache, fixed 8192-point subset per object."""

    def __init__(self, root, image_dir, ids, targets, num_points=8192, seed=42):
        self.root, self.image_dir = Path(root), Path(image_dir)
        self.ids, self.targets = list(ids), np.asarray(targets, dtype=np.float32)
        self.num_points, self.seed = num_points, seed
        self.legacy = (
            self.root / "outputs/geometry_cache/surface_32768_03f72bbb541d/train"
        )
        self.cache = self.root / "outputs/final/surface_32768_seed42"
        self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.ids[index]
        path = self.legacy / f"{item}.npz"
        if not path.exists():
            path = self.cache / f"{item}.npz"
        if not path.exists():
            sample = sample_mesh_surface(
                self.image_dir / f"{item}.npz", item, num_points=32768, seed=42
            )
            np.savez_compressed(
                path,
                points=sample["points"],
                normals=sample["normals"],
                face_ids=sample["face_ids"],
            )
        with np.load(path) as sample:
            indices = np.random.default_rng(
                item_sampling_seed(item, self.seed + 1000)
            ).permutation(len(sample["points"]))[: self.num_points]
            features = np.concatenate(
                [sample["points"][indices], sample["normals"][indices]], axis=1
            )
        return torch.from_numpy(features.copy()), torch.from_numpy(self.targets[index])
