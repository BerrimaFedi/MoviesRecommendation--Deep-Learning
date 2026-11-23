"""Gèle (ou re-gèle) la spécification du découpage : configs/split_spec.json, versionné par Git.

  python -m src.splitting.freeze            # gèle sur les notes actuelles (toutes sources présentes)
  python -m src.splitting.freeze --check    # vérifie que la spécification reproduit le mode recompute

Re-geler change la base d'évaluation : à faire volontairement (avant un ré-entraînement), puis committer
le fichier pour que l'expérience reste traçable. Lancer ensuite `dvc repro`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common.config import load_params
from ..common.io import read_json, write_json
from .split import assign, compute_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = load_params(a.params)
    interim = Path(p["paths"]["interim"])
    movies = pd.read_parquet(interim / "movies_aligned.parquet")
    ratings = pd.read_parquet(interim / "ratings.parquet")
    path = Path(p["split"]["frozen_spec"])
    if a.check:
        spec = read_json(path)
        frozen = ratings[ratings["source"].isin(spec["sources"])]
        same = assign(frozen, spec)[0]["split"].equals(assign(frozen, compute_spec(p, frozen, movies))[0]["split"])
        print(f"{path} : {'identique au mode recompute' if same else 'DIFFÉRENT du mode recompute'}")
        raise SystemExit(0 if same else 1)
    spec = compute_spec(p, ratings, movies)
    write_json(path, spec)
    print(f"{path} : sources {spec['sources']}, {spec['n_ratings']} notes, {len(spec['cold_item_ids'])} films froids, "
          f"train ≤ {spec['train_end_ts']}, val ≤ {spec['val_end_ts']}")


if __name__ == "__main__":
    main()
