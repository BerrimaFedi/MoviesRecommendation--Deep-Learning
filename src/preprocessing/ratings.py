"""Étape `preprocess_ratings` : interactions utilisateur × film (MovieLens + inbox externe).

Sortie : interim/ratings.parquet — user_id, movie_id, rating, timestamp, source. Rejets comptés par motif :
doublon (user, film) -> on garde la note la plus récente ; film hors catalogue ; note hors domaine.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Settings
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from ..ingestion import movielens


def build_ratings(params: dict[str, Any], catalog: pd.DataFrame,
                  external: pd.DataFrame | None = None, external_movies: pd.DataFrame | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Notes MovieLens + notes externes (colonne `source`). Une note externe remplace la note MovieLens
    de la même paire (utilisateur, film) : elle est placée après dans l'ordre de dédoublonnage."""
    cfg = Settings.from_params(params)
    rp = params["preprocessing"]["ratings"]
    _, _, r = movielens.load(cfg.movielens_dir)
    r = r.sort_values("timestamp", kind="stable").assign(source="movielens")
    n_ext = 0
    if external is not None and len(external):
        ext = external.rename(columns={"user_id": "userId", "movie_id": "movieId"}).assign(source="external")
        n_ext = len(ext)
        r = pd.concat([r, ext.astype(r.dtypes.to_dict())], ignore_index=True)
    n0 = len(r)
    dup = r.duplicated(["userId", "movieId"], keep="last")
    r = r[~dup]
    known = set(catalog["movie_id"]) | (set(external_movies["movie_id"].astype(int))
                                        if external_movies is not None and len(external_movies) else set())
    in_cat = r["movieId"].isin(known)
    on_grid = np.isclose((r["rating"] - rp["min"]) / rp["step"], np.round((r["rating"] - rp["min"]) / rp["step"]))
    in_dom = r["rating"].between(rp["min"], rp["max"]) & on_grid
    out = r[in_cat & in_dom].rename(columns={"userId": "user_id", "movieId": "movie_id"})
    out = out.sort_values(["timestamp", "user_id", "movie_id"], kind="stable").reset_index(drop=True)
    info = {
        "ratings_raw": n0, "ratings_external": n_ext, "ratings_kept": len(out),
        "rejected_duplicate_pair": int(dup.sum()),
        "rejected_movie_not_in_catalog": int((~in_cat).sum()),
        "rejected_rating_out_of_domain": int((in_cat & ~in_dom).sum()),
        "rejection_rate": float(1 - len(out) / n0) if n0 else None,
    }
    return out, info


def stage(ctx: StageContext) -> None:
    catalog = pd.read_parquet(ctx.interim / "catalog.parquet")
    ext = ctx.interim / "external"
    out, info = build_ratings(ctx.params, catalog, pd.read_parquet(ext / "ratings.parquet"),
                              pd.read_parquet(ext / "movies.parquet"))
    write_parquet(out, ctx.interim / "ratings.parquet")
    for k, v in info.items():
        ctx.metric(k, v)


if __name__ == "__main__":
    main_for("preprocess_ratings", stage, "Nettoie les notes")
