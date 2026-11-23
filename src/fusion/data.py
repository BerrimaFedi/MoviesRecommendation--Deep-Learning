"""Pont entre les sorties du pipeline (features/items.npz, processed/ratings.parquet) et PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..common.io import read_json


@dataclass
class ItemStore:
    movie_id: np.ndarray
    text: np.ndarray
    image: np.ndarray
    meta: np.ndarray
    mask: np.ndarray
    dims: dict[str, int]

    @classmethod
    def load(cls, features_dir: Path) -> "ItemStore":
        z = np.load(features_dir / "items.npz")
        info = read_json(features_dir / "features.json")
        return cls(z["movie_id"], z["text"], z["image"], z["meta"], z["mask"], info["dims"])


class RatingBatches:
    """Indexation : utilisateurs et films vus à l'entraînement reçoivent un indice ≥ 1, les autres 0
    (inconnu). Un film froid n'a donc jamais d'embedding d'identifiant : seul son contenu le décrit."""

    def __init__(self, store: ItemStore, ratings: pd.DataFrame) -> None:
        self.store = store
        self.ratings = ratings
        train = ratings[ratings["split"] == "train"]
        self.user_index = {u: i + 1 for i, u in enumerate(sorted(train["user_id"].unique()))}
        self.item_index = {m: i + 1 for i, m in enumerate(sorted(train["movie_id"].unique()))}
        self.row = {int(m): i for i, m in enumerate(store.movie_id)}
        self.global_mean = float(train["rating"].mean())

    @property
    def n_users(self) -> int:
        return len(self.user_index) + 1

    @property
    def n_items(self) -> int:
        return len(self.item_index) + 1

    def batch(self, split: str, size: int, seed: int = 0) -> dict[str, Tensor]:
        r = self.ratings[self.ratings["split"] == split]
        r = r.sample(n=min(size, len(r)), random_state=seed)
        rows = np.array([self.row[int(m)] for m in r["movie_id"]])
        s = self.store
        return {
            "text": torch.from_numpy(s.text[rows]).float(),
            "image": torch.from_numpy(s.image[rows]).float(),
            "meta": torch.from_numpy(s.meta[rows]).float(),
            "mask": torch.from_numpy(s.mask[rows]).float(),
            "item_idx": torch.tensor([self.item_index.get(int(m), 0) for m in r["movie_id"]], dtype=torch.long),
            "user_idx": torch.tensor([self.user_index.get(int(u), 0) for u in r["user_id"]], dtype=torch.long),
            "rating": torch.tensor(r["rating"].to_numpy(), dtype=torch.float32),
        }
