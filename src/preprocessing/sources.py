"""Lecture des sources brutes TMDB / Kaggle sous forme de tables, partagée par les étapes de prétraitement."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Settings
from ..ingestion import kaggle, tmdb
from .normalize import clean_text, parse_tmdb_genres, to_float

TMDB_COLS = ["tmdb_id", "tmdb_fetched_at", "t_title", "t_overview", "t_tagline", "genres_tmdb",
             "t_runtime", "t_popularity", "t_vote_average", "t_vote_count", "t_language",
             "t_release_date", "t_poster_path", "t_adult"]
KAGGLE_COLS = ["tmdb_id", "k_overview", "k_tagline", "genres_kaggle", "k_runtime", "k_language",
               "k_release_date", "k_poster_path"]


def coalesce(primary: pd.Series, fallback: pd.Series, names: tuple[str, str] = ("tmdb", "kaggle")):
    """Valeur principale, sinon secours ; renvoie aussi la provenance (traçabilité par cellule)."""
    value = primary.where(primary.notna(), fallback)
    src = pd.Series(np.where(primary.notna(), names[0], np.where(fallback.notna(), names[1], None)),
                    index=primary.index, dtype=object)
    return value, src


def tmdb_frame(cfg: Settings, unmapped: dict[str, int] | None = None) -> tuple[pd.DataFrame, int]:
    rows = []
    for r in tmdb.cache_to_records(cfg.tmdb_cache):
        rows.append({
            "tmdb_id": r["_tmdb_id"], "tmdb_fetched_at": r["_fetched_at"],
            "t_title": clean_text(r.get("title")), "t_overview": clean_text(r.get("overview")),
            "t_tagline": clean_text(r.get("tagline")),
            "genres_tmdb": parse_tmdb_genres(r.get("genres"), unmapped),
            "t_runtime": to_float(r.get("runtime")), "t_popularity": to_float(r.get("popularity")),
            "t_vote_average": to_float(r.get("vote_average")), "t_vote_count": to_float(r.get("vote_count")),
            "t_language": r.get("original_language"), "t_release_date": r.get("release_date"),
            "t_poster_path": r.get("poster_path"), "t_adult": r.get("adult"),
        })
    df = pd.DataFrame(rows, columns=TMDB_COLS)
    df["tmdb_id"] = df["tmdb_id"].astype("Int64")
    n404 = sum(1 for rec in tmdb.load_cache(cfg.tmdb_cache).values() if rec.get("status") == 404)
    return df, n404


def kaggle_frame(cfg: Settings, enabled: bool) -> pd.DataFrame:
    k = kaggle.load(cfg) if enabled else None
    if k is None:
        out = pd.DataFrame(columns=KAGGLE_COLS)
    else:
        out = pd.DataFrame({
            "tmdb_id": k["tmdb_id"], "k_overview": k["overview"], "k_tagline": k["tagline"],
            "genres_kaggle": k["genres"], "k_runtime": k["runtime"], "k_language": k["original_language"],
            "k_release_date": k["release_date"], "k_poster_path": k["poster_path"],
        })
    out["tmdb_id"] = out["tmdb_id"].astype("Int64")
    return out


def catalog_keys(interim_catalog: pd.DataFrame) -> pd.DataFrame:
    """tmdb_id distincts du catalogue (clé de jointure de toutes les modalités de contenu)."""
    return interim_catalog[["tmdb_id"]].dropna().drop_duplicates().reset_index(drop=True)


def restrict(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return keys.merge(df, on="tmdb_id", how="left")


def kaggle_enabled(params: dict[str, Any], cfg: Settings) -> bool:
    return bool(params["kaggle"]["enabled"]) and (cfg.kaggle_dir / cfg.kaggle_file).exists()
