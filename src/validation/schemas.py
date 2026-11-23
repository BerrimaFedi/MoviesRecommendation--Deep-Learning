"""Schémas déclaratifs des tables (colonnes, types, nullabilité, domaines, unicité)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..preprocessing.normalize import genre_columns
from .checks import Suite


@dataclass(frozen=True)
class Col:
    kind: str                    # int | float | str | bool | datetime | list | flag
    nullable: bool = True
    min: float | None = None
    max: float | None = None
    allowed: tuple | None = None
    unique: bool = False


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind == "int":
        return pdt.is_integer_dtype(s)
    if kind == "float":
        return pdt.is_float_dtype(s) or pdt.is_integer_dtype(s)
    if kind == "str":
        return pdt.is_object_dtype(s) or pdt.is_string_dtype(s)
    if kind == "bool":
        return pdt.is_bool_dtype(s)
    if kind == "datetime":
        return pdt.is_datetime64_any_dtype(s)
    if kind == "flag":
        return pdt.is_integer_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}
    if kind == "list":
        sample = s.dropna().head(50)
        return all(isinstance(v, (list, tuple, np.ndarray)) for v in sample)
    raise ValueError(kind)


def check_schema(suite: Suite, df: pd.DataFrame, table: str, spec: dict[str, Col]) -> None:
    missing = [c for c in spec if c not in df.columns]
    suite.check(f"{table}: colonnes attendues présentes", not missing, category="schema",
                observed=missing or "toutes", threshold=list(spec))
    for col, c in spec.items():
        if col not in df.columns:
            continue
        s = df[col]
        suite.check(f"{table}.{col}: type {c.kind}", _kind_ok(s, c.kind), category="schema",
                    observed=str(s.dtype), threshold=c.kind)
        if not c.nullable:
            suite.rate_at_most(f"{table}.{col}: non nul", s.isna(), max_rate=0.0, category="schema")
        num = pd.to_numeric(s, errors="coerce") if c.kind in ("int", "float", "flag") else None
        if num is not None and c.min is not None:
            suite.rate_at_most(f"{table}.{col} ≥ {c.min}", num < c.min, max_rate=0.0, category="schema")
        if num is not None and c.max is not None:
            suite.rate_at_most(f"{table}.{col} ≤ {c.max}", num > c.max, max_rate=0.0, category="schema")
        if c.allowed is not None:
            bad = ~s.dropna().isin(c.allowed)
            suite.rate_at_most(f"{table}.{col} ∈ domaine", bad, max_rate=0.0, category="schema",
                               detail=f"valeurs hors domaine : {sorted(map(str, s.dropna()[bad].unique()))[:5]}")
        if c.unique:
            suite.rate_at_most(f"{table}.{col}: unique", s.dropna().duplicated(), max_rate=0.0,
                               category="duplicates")


# --- sources brutes --------------------------------------------------------------------------------
RAW_ML_MOVIES = {"movieId": Col("int", False, min=1, unique=True), "title": Col("str", False),
                 "genres": Col("str", False)}
RAW_ML_LINKS = {"movieId": Col("int", False, min=1, unique=True), "imdbId": Col("int"), "tmdbId": Col("int")}


def raw_ratings_spec(p: dict[str, Any]) -> dict[str, Col]:
    rp = p["preprocessing"]["ratings"]
    return {"userId": Col("int", False, min=1), "movieId": Col("int", False, min=1),
            "rating": Col("float", False, min=rp["min"], max=rp["max"]),
            "timestamp": Col("int", False, min=789652009)}  # 1995-01-09 : début de MovieLens


KAGGLE_REQUIRED = ["id", "overview", "genres", "runtime", "original_language", "release_date", "poster_path"]
TMDB_REQUIRED_KEYS = ["id", "title", "overview", "genres", "runtime", "original_language", "release_date",
                      "poster_path", "popularity", "vote_count"]

# --- tables finales ----------------------------------------------------------------------------------
SPLITS = ("train", "val", "test", "cold_item_test", "new")


def movies_spec(p: dict[str, Any]) -> dict[str, Col]:
    spec = {
        "movie_id": Col("int", False, min=1, unique=True), "tmdb_id": Col("int"), "imdb_tt": Col("str"),
        "title": Col("str", False), "overview": Col("str"), "overview_source": Col("str", allowed=("tmdb", "kaggle", "external")),
        "poster_file": Col("str"), "poster_w": Col("float", min=p["preprocessing"]["image"]["min_side"]),
        "poster_h": Col("float", min=p["preprocessing"]["image"]["min_side"]),
        "release_date": Col("datetime"), "release_year": Col("int"), "original_language": Col("str", False),
        "lang_cat": Col("str", False), "runtime_min": Col("float", min=p["preprocessing"]["meta"]["runtime_min"]),
        "popularity": Col("float", min=0), "vote_average": Col("float", min=0, max=10), "vote_count": Col("float", min=0),
        "genres_ml": Col("list", False), "genres_unified": Col("list", False),
        "genre_agreement": Col("float", min=0, max=1), "is_imax": Col("flag", False),
        "has_overview": Col("flag", False), "has_poster": Col("flag", False), "has_meta": Col("flag", False),
        "in_tmdb": Col("flag", False), "in_kaggle": Col("flag", False),
        "n_modalities": Col("int", False, min=0, max=3), "all_modalities": Col("flag", False),
        "modality_mask": Col("str", False), "source": Col("str", False, allowed=("movielens", "external")),
    }
    spec.update({g: Col("flag", False) for g in genre_columns()})
    return spec


def ratings_spec(p: dict[str, Any]) -> dict[str, Col]:
    rp = p["preprocessing"]["ratings"]
    return {"user_id": Col("int", False, min=1), "movie_id": Col("int", False, min=1),
            "rating": Col("float", False, min=rp["min"], max=rp["max"]), "timestamp": Col("int", False, min=0),
            "split": Col("str", False, allowed=SPLITS), "source": Col("str", False, allowed=("movielens", "external")),
            "user_in_train": Col("bool", False),
            "item_in_train": Col("bool", False)}
