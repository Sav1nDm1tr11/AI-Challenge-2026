"""Evaluation tables and explanations for the submitted ensemble."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import image_views
from .metrics import scores
from .pipeline import load_heads


def validation_report(assets, config):
    with np.load(Path(assets) / "predictions.npz", allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    rows = []
    for name in ["calibration", "check"]:
        mask = values[name + "_mask"]
        rows.append(
            {
                "split": name,
                **scores(
                    values["targets"][mask],
                    values["probabilities"][mask],
                    config["thresholds"],
                ),
            }
        )
    mask = values["check_mask"]
    y, p = values["targets"][mask], values["probabilities"][mask]
    predicted = p >= config["thresholds"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y[:, :10], predicted, average=None, zero_division=0
    )
    classes = pd.DataFrame(
        {"precision": precision, "recall": recall, "F1": f1, "support": support},
        index=config["defect_columns"],
    )
    quality = ~predicted.any(1)
    matrix = confusion_matrix(y[:, 10], quality, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    classes.F1.sort_values().plot.barh(ax=axes[0], color="#376f91")
    axes[0].set(xlim=(0, 1), xlabel="F1", title="Дефекты: check")
    axes[1].imshow(matrix, cmap="Blues")
    for (r, c), value in np.ndenumerate(matrix):
        axes[1].text(c, r, str(value), ha="center", va="center", fontsize=14)
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["low", "good"],
        yticklabels=["low", "good"],
        xlabel="Предсказание",
        ylabel="Разметка",
        title="Качество: check",
    )
    fig.tight_layout()
    return pd.DataFrame(rows).set_index("split"), classes, fig, values


def plot_training_curves(root, seeds):
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed in seeds:
        history = pd.read_csv(Path(root) / "reports" / f"train_seed{seed}.csv")
        ax.plot(history.epoch, history.train_loss, marker="o", label=f"Seed {seed}")
    ax.set(xlabel="Эпоха", ylabel="BCE", title="Обучение трёх финальных голов")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def explain_object(store, item, image_dir, assets, config, device, defect, truth=None):
    row = store.rows[item]
    patches = torch.from_numpy(np.array(store.patches[row], dtype=np.float32))[None].to(
        device
    )
    heads = load_heads(assets, config, device)
    for head in heads:
        head.requires_grad_(False)
    patches.requires_grad_(True)
    outputs = [head(patches, return_attention=True) for head in heads]
    j = config["defect_columns"].index(defect)
    probability = torch.stack([out[0].sigmoid() for out in outputs]).mean(0)[0, j]
    gradient = torch.autograd.grad(probability, patches)[0]
    attribution = (gradient * patches).sum(-1).detach().cpu().numpy()[0]
    attention = (
        torch.stack([out[1] for out in outputs]).mean(0)[0, :, j].detach().cpu().numpy()
    )
    side = config["resolution"] // config["patch_size"]
    attention = attention.reshape(6, side, side)
    attribution = attribution.reshape(6, side, side)
    with torch.no_grad():
        deltas = []
        for view in range(6):
            altered = patches.detach().clone()
            altered[:, view] = altered[:, [v for v in range(6) if v != view]].mean(1)
            changed = torch.stack([head(altered).sigmoid() for head in heads]).mean(0)[
                0, j
            ]
            deltas.append(float(probability.detach() - changed))
    images = image_views(Path(image_dir) / f"{item}.png", config)
    mean = torch.tensor(config["image_mean"])[:, None, None]
    std = torch.tensor(config["image_std"])[:, None, None]
    rgb = (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    bound = max(float(np.quantile(np.abs(attribution), 0.99)), 1e-12)
    for view in range(6):
        axes[0, view].imshow(rgb[view])
        axes[0, view].set_title(f"Вид {view + 1}: Δp={deltas[view]:+.3f}", fontsize=9)
        axes[1, view].imshow(rgb[view])
        axes[1, view].imshow(
            attention[view],
            extent=(0, 336, 336, 0),
            alpha=0.5,
            cmap="magma",
            vmin=0,
            vmax=float(attention.max()),
        )
        axes[2, view].imshow(
            attribution[view], cmap="coolwarm", vmin=-bound, vmax=bound
        )
        for r in range(3):
            axes[r, view].set_xticks([])
            axes[r, view].set_yticks([])
    axes[0, 0].set_ylabel("Рендер")
    axes[1, 0].set_ylabel("Attention")
    axes[2, 0].set_ylabel("Gradient × token")
    label = "нет разметки" if truth is None else f"разметка={int(truth)}"
    fig.suptitle(
        f"{item}\n{defect}: p={float(probability.detach()):.3f}, порог={config['thresholds'][j]:.3f}; {label}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
