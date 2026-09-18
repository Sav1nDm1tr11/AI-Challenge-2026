"""Six-view preprocessing and resumable, disk-backed patch features."""

import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def image_views(path, config):
    with Image.open(path) as image:
        image = image.convert("RGB")
    width, height = image.size
    if width % 3 or height % 2 or width // 3 != height // 2:
        raise ValueError(f"Expected a 3×2 grid of square views: {path}")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["resolution"], config["resolution"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(config["image_mean"], config["image_std"]),
        ]
    )
    return torch.stack(
        [
            transform(
                image.crop(
                    (
                        c * width // 3,
                        r * height // 2,
                        (c + 1) * width // 3,
                        (r + 1) * height // 2,
                    )
                )
            )
            for r in range(2)
            for c in range(3)
        ]
    )


def load_backbone(assets, config, device):
    model = timm.create_model(
        config["model_name"],
        pretrained=False,
        num_classes=0,
        img_size=224,
        dynamic_img_size=True,
    )
    model.load_state_dict(
        torch.load(Path(assets) / "backbone.pt", map_location="cpu", weights_only=True)
    )
    return model.to(device).eval().requires_grad_(False)


class PatchStore:
    def __init__(self, folder, ids, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ids = list(ids)
        self.rows = {item: index for index, item in enumerate(ids)}
        self.config = config
        self.tokens = (config["resolution"] // config["patch_size"]) ** 2
        metadata = {
            "ids": self.ids,
            "resolution": config["resolution"],
            "source_sha256": config["source_checkpoint_sha256"],
            "image_mean": config["image_mean"],
            "image_std": config["image_std"],
            "dtype": "float16",
            "timm": timm.__version__,
        }
        manifest = self.folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != metadata:
            raise ValueError(
                "Feature cache belongs to another configuration or ID order."
            )
        manifest.write_text(json.dumps(metadata, indent=2))
        shape = (len(ids), config["views"], self.tokens, config["feature_dim"])
        self.patches = self._array("patch.npy", shape, np.float16)
        self.ready = self._array("ready.npy", (len(ids),), np.bool_)

    def _array(self, name, shape, dtype):
        path = self.folder / name
        existed = path.exists()
        array = np.lib.format.open_memmap(
            path, mode="r+" if existed else "w+", dtype=dtype, shape=shape
        )
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Invalid cache array: {path}")
        if not existed and dtype == np.bool_:
            array[:] = False
            array.flush()
        return array

    @classmethod
    def open_legacy(cls, folder, ids, config):
        """Reuse the verified research cache locally without copying tens of GB."""
        folder = Path(folder)
        metadata = json.loads((folder / "manifest.json").read_text())
        if (
            metadata["source_sha256"] != config["source_checkpoint_sha256"]
            or metadata["resolution"] != config["resolution"]
            or metadata["normalization"] != config["image_mean"]
            or metadata["std"] != config["image_std"]
        ):
            raise ValueError("Legacy cache metadata mismatch")
        store = object.__new__(cls)
        store.folder, store.ids, store.config = folder, metadata["ids"], config
        store.rows = {item: index for index, item in enumerate(store.ids)}
        if not set(ids).issubset(store.rows):
            raise ValueError("IDs missing from legacy cache")
        store.patches = np.load(folder / "patch.npy", mmap_mode="r+")
        store.ready = np.load(folder / "ready.npy", mmap_mode="r+")
        store.tokens = (config["resolution"] // config["patch_size"]) ** 2
        return store

    def ensure(self, ids, image_dir, assets, device):
        pending = [item for item in ids if not self.ready[self.rows[item]]]
        if not pending:
            return
        encoder = load_backbone(assets, self.config, device)
        with torch.inference_mode():
            for item in tqdm(pending, desc="DINOv2 patch features"):
                images = image_views(Path(image_dir) / f"{item}.png", self.config).to(
                    device
                )
                tokens = torch.cat(
                    [
                        encoder.forward_features(images[i : i + 3])
                        for i in range(0, 6, 3)
                    ]
                )
                patches = tokens[:, encoder.num_prefix_tokens :]
                self.patches[self.rows[item]] = patches.cpu().numpy().astype(np.float16)
                self.patches.flush()
                self.ready[self.rows[item]] = True
                self.ready.flush()
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()


class PatchDataset(Dataset):
    def __init__(self, store, ids, targets=None):
        self.store, self.ids, self.targets = store, list(ids), targets

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = self.store.rows[self.ids[index]]
        if not self.store.ready[row]:
            raise RuntimeError(
                "Extract features before starting training or inference."
            )
        patches = torch.from_numpy(np.array(self.store.patches[row], dtype=np.float32))
        return (
            patches
            if self.targets is None
            else (patches, torch.as_tensor(self.targets[index], dtype=torch.float32))
        )
