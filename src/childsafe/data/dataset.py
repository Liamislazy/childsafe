"""Torch dataset and transforms for Module A."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from childsafe.data.corruptions import FixedCorruption, RobustAug

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Letterbox:
    """Keep the full thin object visible rather than center-cropping its endpoints."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=(124, 116, 104))


def train_transform(size: int, robust: bool = False, preprocessing: str = "center_crop"):
    if preprocessing not in ('center_crop','letterbox'):
        raise ValueError('unsupported checkpoint preprocessing')
    tf = [Letterbox(size) if preprocessing == "letterbox" else T.RandomResizedCrop(size, scale=(0.6, 1.0)),
          T.RandomHorizontalFlip(),
          T.ColorJitter(0.3, 0.3, 0.2, 0.02)]
    if robust:
        tf.append(RobustAug(p=0.5))
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if robust:
        tf.append(T.RandomErasing(p=0.4, scale=(0.02, 0.25), value="random"))
    return T.Compose(tf)


def eval_transform(size: int, corruption: tuple[str, int] | None = None, preprocessing: str = "center_crop"):
    if preprocessing not in ('center_crop','letterbox'):
        raise ValueError('unsupported checkpoint preprocessing')
    tf = [Letterbox(size)] if preprocessing == "letterbox" else [T.Resize(int(size * 1.14)), T.CenterCrop(size)]
    if corruption and corruption[1] > 0:
        tf.append(FixedCorruption(*corruption))
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(tf)


def image_paths(df: pd.DataFrame, root: str | Path | dict) -> list[Path]:
    """root: one folder, or {source: folder} when df has a `source` column (ChildSUn + extra sources)."""
    if isinstance(root, dict):
        src = df["source"] if "source" in df else ["childsun"] * len(df)
        return [Path(root.get(s, root["childsun"])) / p for s, p in zip(src, df["path"])]
    return [Path(root) / p for p in df["path"]]


class CatalogDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path | dict, classes: list[str], transform):
        self.paths = image_paths(df, root)
        idx = {c: i for i, c in enumerate(classes)}
        self.targets = torch.tensor([idx[c] for c in df["category"]])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]
