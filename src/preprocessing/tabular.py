"""Étape `preprocess_meta` : modalité tabulaire (métadonnées TMDB, secours Kaggle).

Sortie : interim/meta.parquet, une ligne par tmdb_id du catalogue — genres_tmdb (canoniques),
original_language (ISO 639-1), release_date, runtime_min, popularity, vote_average, vote_count,
log_popularity, log_vote_count, in_tmdb, in_kaggle, tmdb_fetched_at.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Settings
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from .normalize import normalize_language
from .sources import catalog_keys, coalesce, kaggle_enabled, kaggle_frame, restrict, tmdb_frame


def build_meta(params: dict[str, Any], catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = Settings.from_params(params)
    mp = params["preprocessing"]["meta"]
    unmapped: dict[str, int] = {}
    t, n404 = tmdb_frame(cfg, unmapped)
    k = kaggle_frame(cfg, kaggle_enabled(params, cfg))
    df = restrict(t, catalog_keys(catalog)).merge(k, on="tmdb_id", how="left")
    df["in_tmdb"] = df["tmdb_fetched_at"].notna().astype("int8")
    df["in_kaggle"] = (df["k_language"].notna() | df["k_overview"].notna()).astype("int8")

    # genres côté TMDB : liste TMDB si non vide, sinon Kaggle (même vocabulaire, instantané 2017)
    df["genres_tmdb"] = [a if isinstance(a, list) and a else (b if isinstance(b, list) else [])
                         for a, b in zip(df["genres_tmdb"], df["genres_kaggle"])]

    lang_t = df["t_language"].where(df["t_language"].fillna("").astype(str).str.strip() != "")
    lang, _ = coalesce(lang_t, df["k_language"])
    df["original_language"] = lang.map(normalize_language)

    rd, _ = coalesce(df["t_release_date"].replace("", np.nan), df["k_release_date"].replace("", np.nan))
    df["release_date"] = pd.to_datetime(rd, errors="coerce")

    runtime, _ = coalesce(df["t_runtime"], df["k_runtime"])
    runtime = pd.to_numeric(runtime, errors="coerce")
    df["runtime_min"] = runtime.where(runtime >= mp["runtime_min"])  # 0 = inconnu chez TMDB
    # popularité et votes : TMDB uniquement (instantané daté `tmdb_fetched_at`) ; pas de mélange avec
    # l'instantané Kaggle 2017 dont l'échelle et la date ne sont pas comparables.
    df["popularity"] = pd.to_numeric(df["t_popularity"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["t_vote_count"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["t_vote_average"], errors="coerce").where(df["vote_count"] > 0)
    df["log_popularity"] = np.log1p(df["popularity"])
    df["log_vote_count"] = np.log1p(df["vote_count"])
    out = df[["tmdb_id", "genres_tmdb", "original_language", "release_date", "runtime_min", "popularity",
              "vote_average", "vote_count", "log_popularity", "log_vote_count", "in_tmdb", "in_kaggle",
              "tmdb_fetched_at"]]
    info = {
        "rows": len(out), "tmdb_404": n404, "found_in_tmdb": int(out["in_tmdb"].sum()),
        "found_in_kaggle": int(out["in_kaggle"].sum()), "unmapped_tmdb_genres": unmapped,
        "runtime_above_max": int((out["runtime_min"] > mp["runtime_max"]).sum()),
        "runtime_missing": int(out["runtime_min"].isna().sum()),
    }
    return out, info


def stage(ctx: StageContext) -> None:
    catalog = pd.read_parquet(ctx.interim / "catalog.parquet")
    out, info = build_meta(ctx.params, catalog)
    write_parquet(out, ctx.interim / "meta.parquet")
    for k, v in info.items():
        ctx.metric(k, v)


if __name__ == "__main__":
    main_for("preprocess_meta", stage, "Prépare la modalité tabulaire")
