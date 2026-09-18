"""Six-epoch frozen-head training; original split, seeds and accumulation."""

import gc
import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PatchDataset
from .model import PatchClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_ensemble(store, ids, targets, config, output, device):
    """Write new heads separately; never overwrite the submitted weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "config": config,
        "ids": list(ids),
        "device": str(device),
        "targets_sha256": hashlib.sha256(
            np.asarray(targets, dtype=np.float32).tobytes()
        ).hexdigest(),
    }
    manifest = output / "training_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError(
            "Use a new output directory for another training configuration."
        )
    manifest.write_text(json.dumps(signature, indent=2))
    for seed in config["seeds"]:
        final = output / f"head_seed{seed}.pt"
        resume = output / f"resume_seed{seed}.pt"
        if final.exists():
            print(f"""Seed {seed}: сохранённые веса уже есть.""")
            continue
        seed_everything(seed)
        model = PatchClassifier(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            PatchDataset(store, ids, targets),
            batch_size=config["microbatch"],
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        history, start = [], 1
        if resume.exists():
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if (
                state["ids"] != list(ids)
                or state["config"] != config
                or state["device"] != str(device)
            ):
                raise ValueError("Resume belongs to a different training setup.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            generator.set_state(state["loader_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device.type == "mps":
                torch.mps.set_rng_state(state["device_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(state["device_rng"])
            history, start = state["history"], state["epoch"] + 1
        accumulation = config["effective_batch"] // config["microbatch"]
        for epoch in range(start, config["epochs"] + 1):
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for step, (patches, labels) in enumerate(loader):
                patches, labels = patches.to(device), labels[:, :10].to(device)
                window_start = (step // accumulation) * config["effective_batch"]
                window_size = min(config["effective_batch"], len(ids) - window_start)
                amp = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(patches).float(), labels.float()
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss * len(patches) / window_size).backward()
                total += float(loss.detach()) * len(patches)
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            row = {
                "epoch": epoch,
                "train_loss": total / len(ids),
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(
                output / f"history_seed{seed}.csv", index=False
            )
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "config": config,
                "ids": list(ids),
                "device": str(device),
                "torch_rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
            }
            if device.type == "mps":
                state["device_rng"] = torch.mps.get_rng_state()
            if device.type == "cuda":
                state["device_rng"] = torch.cuda.get_rng_state_all()
            torch.save(state, resume.with_suffix(".tmp"))
            resume.with_suffix(".tmp").replace(resume)
            print(
                f"""Seed {seed} | эпоха {epoch}/{config["epochs"]}
Train loss: {row["train_loss"]:.5f} | время: {row["seconds"]:.1f} с
""",
                flush=True,
            )
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, final)
        del model, optimizer, scaler
        gc.collect()
