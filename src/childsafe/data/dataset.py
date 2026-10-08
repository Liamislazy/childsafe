"""Torch dataset and transforms for Module A."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from childsafe.data.corruptions import FixedCorruption, RobustAug

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(size: int, robust: bool = False):
    tf = [T.RandomResizedCrop(size, scale=(0.6, 1.0)),
          T.RandomHorizontalFlip(),
          T.ColorJitter(0.3, 0.3, 0.2, 0.02)]
    if robust:
        tf.append(RobustAug(p=0.5))
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if robust:
        tf.append(T.RandomErasing(p=0.4, scale=(0.02, 0.25), value="random"))
    return T.Compose(tf)


def eval_transform(size: int, corruption: tuple[str, int] | None = None):
    tf = [T.Resize(int(size * 1.14)), T.CenterCrop(size)]
    if corruption and corruption[1] > 0:
        tf.append(FixedCorruption(*corruption))
    tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(tf)


def image_paths(df: pd.DataFrame, root: str | Path | dict) -> list[Path]:
    """root: one folder, or {source: folder} when df has a `source` column (ChildSUn + extra sources)."""
    if isinstance(root, dict):
        src = df["source"] if "source" in df else ["childsun"] * len(df)
        return [Path(root[s]) / p for s, p in zip(src, df["path"])]
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
