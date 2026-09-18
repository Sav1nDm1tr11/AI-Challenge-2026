"""Portable loading, ensemble inference and submission export."""

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PatchDataset, PatchStore
from .model import PatchClassifier


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def download_bundle(url, destination):
    """Download an explicitly configured ZIP; reject members outside destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "download.tmp.zip"
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not (destination / member).resolve().is_relative_to(destination):
                raise ValueError("Unsafe ZIP member")
        bundle.extractall(destination)
    archive.unlink()


def load_project(root):
    root = Path(root).resolve()
    config = json.loads((root / "configs/final.json").read_text())
    assets = root / "artifacts/final"
    if not (assets / "backbone.pt").exists() and os.getenv("MESH_ASSETS_URL"):
        download_bundle(os.environ["MESH_ASSETS_URL"], root)
    if not (assets / "backbone.pt").exists():
        raise FileNotFoundError(
            "Unpack final_assets.zip in the project or set MESH_ASSETS_URL."
        )
    manifest = json.loads((root / "configs/assets.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(assets / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    data_root = Path(os.environ.get("MESH_DATA_ROOT", root)).resolve()
    if not (data_root / "test.csv").exists() and os.getenv("MESH_DATA_URL"):
        download_bundle(os.environ["MESH_DATA_URL"], data_root)
    return config, assets, data_root


def feature_store(root, ids, config, split):
    root = Path(root)
    source = config["source_checkpoint_sha256"][:12]
    legacy = (
        root / "outputs/research_lab/next_steps/test_cache/tokens"
        if split == "test"
        else root / "outputs/research_lab/tokens"
    ) / f"r336_{source}"
    if (legacy / "manifest.json").exists():
        return PatchStore.open_legacy(legacy, ids, config)
    folder = root / "outputs/final/features" / split
    needed = len(ids) * 6 * 576 * 384 * 2
    folder.mkdir(parents=True, exist_ok=True)
    if (
        not (folder / "patch.npy").exists()
        and shutil.disk_usage(folder).free < needed + 1024**3
    ):
        raise RuntimeError(
            f"Need approximately {needed / 1024**3:.1f} GiB plus 1 GiB free."
        )
    return PatchStore(folder, ids, config)


def load_heads(assets, config, device):
    heads = []
    for seed in config["seeds"]:
        model = PatchClassifier(config)
        model.load_state_dict(
            torch.load(
                Path(assets) / f"head_seed{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        heads.append(model.to(device).eval())
    return heads


@torch.inference_mode()
def predict_ensemble(store, ids, assets, config, device, batch_size=1):
    models = load_heads(assets, config, device)
    loader = DataLoader(
        PatchDataset(store, ids), batch_size=batch_size, shuffle=False, num_workers=0
    )
    result = []
    for patches in tqdm(loader, desc="Three-seed ensemble"):
        patches = patches.to(device)
        # Each model's sigmoid is rounded to float32 before float64 averaging,
        # matching the original submitted pipeline.
        members = [model(patches).sigmoid().cpu().numpy() for model in models]
        result.append(
            (sum(p.astype(np.float64) for p in members) / len(members)).astype(
                np.float32
            )
        )
    return np.concatenate(result)


def save_submission(ids, probabilities, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(ids), 10) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction array")
    defects = (probabilities >= np.asarray(config["thresholds"])).astype(np.int64)
    frame = pd.DataFrame(defects, columns=config["defect_columns"])
    frame.insert(0, "item_id", list(ids))
    frame["quality"] = (~defects.any(axis=1)).astype(np.int64)
    if not frame.item_id.is_unique:
        raise ValueError("Duplicate test IDs")
    frame.to_csv(path, index=False)
    return frame
