"""Étape `preprocess_catalog` : catalogue MovieLens (films + liens) normalisé.

Sortie : interim/catalog.parquet — movie_id, tmdb_id, imdb_tt, title, title_movielens, year_title,
genres_ml (vocabulaire canonique), is_imax.
"""
from __future__ import annotations

import pandas as pd

from ..common.config import Settings
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from ..ingestion import movielens
from .normalize import imdb_tt, parse_ml_genres, split_title_year


def build_catalog(cfg: Settings) -> tuple[pd.DataFrame, dict]:
    ml_movies, ml_links, _ = movielens.load(cfg.movielens_dir)
    unmapped: dict[str, int] = {}
    base = ml_movies.merge(ml_links, on="movieId", how="left")
    parsed = base["genres"].map(lambda s: parse_ml_genres(s, unmapped))
    base["genres_ml"] = parsed.map(lambda t: t[0])
    base["is_imax"] = parsed.map(lambda t: int(t[1])).astype("int8")
    ty = base["title"].map(split_title_year)
    base["title_clean"] = ty.map(lambda t: t[0])
    base["year_title"] = pd.array(ty.map(lambda t: t[1]).tolist(), dtype="Int64")
    base["imdb_tt"] = base["imdbId"].map(imdb_tt)
    base = base.rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id", "title": "title_movielens",
                                "title_clean": "title"})
    base["tmdb_id"] = base["tmdb_id"].astype("Int64")
    out = base[["movie_id", "tmdb_id", "imdb_tt", "title", "title_movielens", "year_title", "genres_ml",
                "is_imax"]].sort_values("movie_id").reset_index(drop=True)
    info = {
        "movies": len(out),
        "tmdb_id_missing_in_links": int(out["tmdb_id"].isna().sum()),
        "tmdb_id_duplicated_in_links": int(out["tmdb_id"].dropna().duplicated().sum()),
        "unmapped_movielens_genres": unmapped,
        "movies_without_ml_genre": int((out["genres_ml"].map(len) == 0).sum()),
    }
    return out, info


def stage(ctx: StageContext) -> None:
    cfg = Settings.from_params(ctx.params)
    cat, info = build_catalog(cfg)
    write_parquet(cat, ctx.interim / "catalog.parquet")
    for k, v in info.items():
        ctx.metric(k, v)
    ctx.info("catalogue écrit", rows=len(cat))


if __name__ == "__main__":
    main_for("preprocess_catalog", stage, "Normalise le catalogue MovieLens")
