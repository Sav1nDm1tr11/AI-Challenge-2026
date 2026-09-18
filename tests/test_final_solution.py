"""Regression checks live outside the presentation notebook."""

import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mesh_quality.model import PatchClassifier
from mesh_quality.pipeline import save_submission


def test_checkpoint_algebra():
    config = json.loads((ROOT / "configs/final.json").read_text())
    model = PatchClassifier(config).eval()
    weights = torch.load(
        ROOT / "artifacts/final/head_seed42.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(weights)
    patches = torch.randn(2, 6, 8, 384)
    with torch.no_grad():
        tokens = model.projection(patches.flatten(1, 2))
        attention = model.attention(tokens).softmax(1)
        local = torch.einsum("btc,btd->bcd", attention, tokens)
        original_global = torch.zeros(2, 6, 384)
        pooled = torch.cat([original_global.mean(1), original_global.amax(1)], -1)
        original_representation = model.fusion(
            torch.cat([pooled[:, None].expand(-1, 10, -1), local], -1)
        )
        expected = (original_representation * model.defect_head.weight[None]).sum(
            -1
        ) + model.defect_head.bias
        torch.testing.assert_close(model(patches), expected, rtol=0, atol=0)


def test_submission_rule(tmp_path):
    config = json.loads((ROOT / "configs/final.json").read_text())
    p = np.zeros((2, 10), dtype=np.float32)
    p[1, 3] = 1.0
    frame = save_submission(["first", "second"], p, config, tmp_path / "submission.csv")
    assert frame.quality.tolist() == [1, 0]
    assert frame.lowpoly.tolist() == [0, 1]
    assert frame.columns.tolist() == ["item_id", *config["defect_columns"], "quality"]


def test_training_smoke(tmp_path):
    from mesh_quality.training import train_ensemble
    from types import SimpleNamespace

    config = json.loads((ROOT / "configs/final.json").read_text())
    config.update(epochs=1, seeds=[42], microbatch=1, effective_batch=4)
    ids = [f"item_{i}" for i in range(5)]
    rng = np.random.default_rng(42)
    patches = rng.normal(size=(5, 6, 8, 384)).astype(np.float16)
    store = SimpleNamespace(
        rows={item: i for i, item in enumerate(ids)},
        ready=np.ones(5, bool),
        patches=patches,
    )
    targets = rng.integers(0, 2, size=(5, 11)).astype(np.float32)
    train_ensemble(store, ids, targets, config, tmp_path, torch.device("cpu"))
    weights = torch.load(
        tmp_path / "head_seed42.pt", map_location="cpu", weights_only=True
    )
    model = PatchClassifier(config)
    model.load_state_dict(weights)
    assert all(torch.isfinite(value).all() for value in weights.values())
