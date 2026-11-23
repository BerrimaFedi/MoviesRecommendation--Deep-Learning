"""Étape `split` : découpage train / val / test / cold_item_test sans fuite, et export final.

- Démarrage à froid strict : cold_item_frac des films (≥ cold_item_min_ratings notes et au moins un
  contenu) sont retirés de train/val/test ; toutes leurs notes vont dans `cold_item_test`.
- Découpage chronologique global des autres notes : 80 % les plus anciennes en train, 10 % val, 10 %
  test ; les égalités de timestamp restent du même côté (aucune note de test antérieure au train).
- Graine fixée (project.seed) : même entrée + même configuration = même découpage.

Deux modes (split.mode) :
- frozen (défaut) : applique configs/split_spec.json, figé et versionné. L'évaluation reste comparable quand
  des données arrivent : les notes d'une source non couverte par la spécification (inbox externe) vont dans
  un segment `new`, sans toucher à train/val/test. `python -m src.splitting.freeze` re-gèle la spécification
  sur toutes les données actuelles (nouvelle base de comparaison, avant un ré-entraînement).
- recompute : tire les films froids et calcule les bornes à chaque exécution (CI, échantillon).

Sorties : processed/movies.parquet, processed/ratings.parquet, processed/manifest.json.
"""
from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.io import read_json, write_json, write_parquet
from ..common.stage import StageContext, main_for

ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."


def compute_spec(params: dict[str, Any], ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    """Tire les films froids et calcule les bornes chronologiques sur `ratings` (mode recompute)."""
    sp = params["split"]
    counts = ratings.groupby("movie_id").size()
    has_content = movies.set_index("movie_id")[["has_overview", "has_poster"]].max(axis=1) > 0
    eligible = sorted(set(counts[counts >= sp["cold_item_min_ratings"]].index) & set(has_content[has_content].index))
    rng = np.random.default_rng(params["project"]["seed"])
    n_cold = int(round(len(eligible) * sp["cold_item_frac"]))
    cold_ids = sorted(int(x) for x in rng.choice(eligible, size=n_cold, replace=False)) if n_cold else []
    warm = ratings[~ratings["movie_id"].isin(cold_ids)].sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
    n = len(warm)
    t_train = int(warm["timestamp"].iloc[max(int(n * sp["train_frac"]) - 1, 0)]) if n else None
    t_val = int(warm["timestamp"].iloc[max(int(n * (sp["train_frac"] + sp["val_frac"])) - 1, 0)]) if n else None
    sources = sorted(ratings["source"].unique()) if "source" in ratings else ["movielens"]
    return {"seed": params["project"]["seed"], "train_frac": sp["train_frac"], "val_frac": sp["val_frac"],
            "cold_item_frac": sp["cold_item_frac"], "cold_item_min_ratings": sp["cold_item_min_ratings"],
            "train_end_ts": t_train, "val_end_ts": t_val, "cold_item_ids": cold_ids, "cold_eligible": len(eligible),
            "sources": sources, "n_ratings": int(len(ratings))}


def assign(ratings: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Applique une spécification : les notes des sources figées suivent les bornes et la liste de films
    froids ; les notes d'autres sources (ex. inbox arrivée après le gel) vont dans le segment `new`."""
    r = ratings.copy()
    source = r["source"] if "source" in r else pd.Series("movielens", index=r.index)
    frozen = source.isin(spec["sources"])
    is_cold = r["movie_id"].isin(spec["cold_item_ids"])
    ts = r["timestamp"]
    r["split"] = "new"
    r.loc[frozen & is_cold, "split"] = "cold_item_test"
    if spec["train_end_ts"] is not None:
        warm = frozen & ~is_cold
        r.loc[warm & (ts <= spec["train_end_ts"]), "split"] = "train"
        r.loc[warm & (ts > spec["train_end_ts"]) & (ts <= spec["val_end_ts"]), "split"] = "val"
        r.loc[warm & (ts > spec["val_end_ts"]), "split"] = "test"
    train = r[r["split"] == "train"]
    r["user_in_train"] = r["user_id"].isin(set(train["user_id"]))
    r["item_in_train"] = r["movie_id"].isin(set(train["movie_id"]))
    r = r.sort_values(["timestamp", "user_id", "movie_id"], kind="stable").reset_index(drop=True)
    vt = r["split"].isin(["val", "test"])
    info: dict[str, Any] = {
        "split_sizes": r["split"].value_counts().to_dict(), "cold_items": len(spec["cold_item_ids"]),
        "cold_eligible": spec.get("cold_eligible"),
        "unseen_item_share_val_test": float(1 - r.loc[vt, "item_in_train"].mean()) if vt.any() else None,
        "new_segment": int((r["split"] == "new").sum()),
    }
    if spec["train_end_ts"] is not None:
        info["train_end"] = datetime.fromtimestamp(spec["train_end_ts"], UTC).isoformat()
        info["val_end"] = datetime.fromtimestamp(spec["val_end_ts"], UTC).isoformat()
    return r, info


def split_ratings(params: dict[str, Any], ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mode recompute : spécification recalculée sur toutes les notes présentes."""
    if "source" not in ratings:
        ratings = ratings.assign(source="movielens")
    return assign(ratings, compute_spec(params, ratings, movies))


def load_spec(params: dict[str, Any]) -> dict[str, Any]:
    path = Path(params["split"]["frozen_spec"])
    if not path.exists():
        raise FileNotFoundError(f"split.mode=frozen mais {path} est absent : lancez `python -m src.splitting.freeze`")
    spec = read_json(path)
    for k in ("seed", "train_frac", "val_frac", "cold_item_frac", "cold_item_min_ratings"):
        if spec[k] != (params["project"]["seed"] if k == "seed" else params["split"][k]):
            raise ValueError(f"{path} a été figé avec {k}={spec[k]} : relancez `python -m src.splitting.freeze`")
    return spec


def stage(ctx: StageContext) -> None:
    movies = pd.read_parquet(ctx.interim / "movies_aligned.parquet")
    ratings = pd.read_parquet(ctx.interim / "ratings.parquet")
    mode = ctx.params["split"]["mode"]
    if mode == "frozen":
        spec = load_spec(ctx.params)
        r, info = assign(ratings, spec)
        missing = set(spec["cold_item_ids"]) - set(movies["movie_id"])
        if missing:
            ctx.warn("films froids de la spécification absents du catalogue", n=len(missing))
    elif mode == "recompute":
        r, info = split_ratings(ctx.params, ratings, movies)
    else:
        raise ValueError(f"split.mode inconnu : {mode}")
    info["mode"] = mode
    write_parquet(movies, ctx.processed / "movies.parquet")
    write_parquet(r, ctx.processed / "ratings.parquet")
    for k, v in info.items():
        ctx.metric(k, v)

    p = ctx.params
    md5 = ctx.paths["raw_movielens"] / "SOURCE_MD5.txt"
    # pas d'horodatage ici : mêmes entrées + même configuration => artefact identique octet pour octet
    # (la date d'exécution est dans reports/stages/split.json)
    write_json(ctx.processed / "manifest.json", {
        "ingestion_mode": p["ingestion"]["mode"], "movielens_version": p["movielens"]["version"],
        "movielens_source_md5": md5.read_text().strip() if md5.exists() else None,
        "tmdb_language": p["tmdb"]["language"], "poster_size": p["tmdb"]["poster_size"],
        "seed": p["project"]["seed"], "split": p["split"],
        "counts": {"movies": len(movies), "ratings": len(r), "users": int(r["user_id"].nunique())},
        "split_info": info, "movies_columns": list(movies.columns), "ratings_columns": list(r.columns),
        "python": platform.python_version(), "pandas": pd.__version__, "attribution": ATTRIBUTION,
    })


if __name__ == "__main__":
    main_for("split", stage, "Découpe les notes sans fuite et exporte les tables finales")
