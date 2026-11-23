"""Étape `align` : appariement des modalités par clé (movie_id ↔ tmdb_id) et masques de présence.

Les trois modalités de contenu (texte, image, tabulaire) sont indexées par tmdb_id ; le catalogue et
les notes par movie_id ; links.csv fait le pont. Les modalités ne sont pas des séquences horodatées :
l'alignement est un appariement échantillon par échantillon, sans synchronisation fine à calculer.
La seule contrainte temporelle (sortie du film ≤ première note) est contrôlée par la validation.

Un film n'est jamais supprimé parce qu'une modalité manque : l'absence est marquée explicitement
(has_overview, has_poster, has_meta, n_modalities, modality_mask « TIM » / « T-M » / …).

Les films apportés par un tiers (inbox, étape `ingest_external`) sont ajoutés avec leur propre contenu
(source = « external ») et passent par les mêmes colonnes dérivées et les mêmes contrôles.

Sortie : interim/movies_aligned.parquet (une ligne par movie_id).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from ..preprocessing.normalize import CANON_GENRES, genre_columns, jaccard, slug

MOVIE_COLUMNS = (
    ["movie_id", "tmdb_id", "imdb_tt", "title", "title_movielens", "title_tmdb", "overview",
     "overview_source", "tagline", "overview_n_chars", "overview_n_words", "poster_path", "poster_path_source",
     "poster_file", "poster_w", "poster_h", "poster_sha1", "release_date", "release_year", "year_title",
     "original_language", "lang_cat", "runtime_min", "popularity", "vote_average", "vote_count",
     "log_popularity", "log_vote_count", "genres_ml", "genres_unified", "genre_agreement", "is_imax"]
    + genre_columns()
    + ["has_overview", "has_poster", "has_meta", "in_tmdb", "in_kaggle", "runtime_missing", "year_missing",
       "snapshot_features_missing", "n_modalities", "all_modalities", "modality_mask", "source", "tmdb_fetched_at"]
)


def as_list(value: Any) -> list:
    """Les listes relues depuis Parquet sont des ndarray ; une jointure manquante donne NaN."""
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def external_rows(ext: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Films de l'inbox au format des colonnes fusionnées (avant les colonnes dérivées)."""
    if ext is None or not len(ext):
        return pd.DataFrame()
    ov = ext["overview"]
    n_chars = ov.fillna("").str.len()
    has_ov = n_chars >= min_len
    has_po = ext["poster_file"].notna()
    meta_present = ext["genres"].map(len).gt(0) | ext["year"].notna() | ext["runtime_min"].notna()
    return pd.DataFrame({
        "movie_id": ext["movie_id"].astype("int64"), "tmdb_id": pd.array([pd.NA] * len(ext), dtype="Int64"),
        "imdb_tt": None, "title": ext["title"], "title_movielens": None, "year_title": ext["year"].astype("Int64"),
        "genres_ml": ext["genres"].map(list), "is_imax": 0,
        "overview": ov, "overview_source": np.where(has_ov, "external", None), "tagline": None, "title_tmdb": None,
        "overview_n_chars": n_chars, "overview_n_words": ov.fillna("").str.split().map(len), "has_overview": has_ov.astype(int),
        "poster_path": None, "poster_path_source": np.where(has_po, "external", None), "poster_file": ext["poster_file"],
        "poster_w": ext["poster_w"], "poster_h": ext["poster_h"], "poster_sha1": ext["poster_sha1"],
        "poster_decode_ok": has_po, "has_poster": has_po.astype(int),
        "genres_tmdb": [[] for _ in range(len(ext))], "original_language": ext["original_language"],
        "release_date": pd.NaT, "runtime_min": ext["runtime_min"], "popularity": np.nan, "vote_average": np.nan,
        "vote_count": np.nan, "log_popularity": np.nan, "log_vote_count": np.nan, "in_tmdb": 0, "in_kaggle": 0,
        "tmdb_fetched_at": None, "source": "external", "_ext_meta": meta_present.astype(int),
    })


def align(params: dict[str, Any], catalog: pd.DataFrame, text: pd.DataFrame, images: pd.DataFrame,
          meta: pd.DataFrame, external: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    for name, df in (("text", text), ("images", images), ("meta", meta)):
        if df["tmdb_id"].dropna().duplicated().any():
            raise ValueError(f"clé tmdb_id non unique dans la modalité {name}")
    df = (catalog.merge(text, on="tmdb_id", how="left", validate="many_to_one")
                 .merge(images, on="tmdb_id", how="left", validate="many_to_one")
                 .merge(meta, on="tmdb_id", how="left", validate="many_to_one"))
    df["source"], df["_ext_meta"] = "movielens", 0
    ext = external_rows(external, params["preprocessing"]["text"]["min_len_chars"])
    if len(ext):
        if ext["movie_id"].isin(df["movie_id"]).any():
            raise ValueError("movie_id externe en collision avec le catalogue MovieLens")
        df = pd.concat([df, ext[df.columns]], ignore_index=True)

    # --- genres : union MovieLens ∪ TMDB(/Kaggle) dans le vocabulaire canonique ------------------
    order = {g: i for i, g in enumerate(CANON_GENRES)}
    tmdb_g = [as_list(g) for g in df["genres_tmdb"]]  # NaN si le film n'a pas de métadonnées
    ml_g = [as_list(g) for g in df["genres_ml"]]
    df["genres_ml"] = ml_g
    df["genres_unified"] = [sorted(set(a) | set(b), key=order.__getitem__) for a, b in zip(ml_g, tmdb_g)]
    df["genre_agreement"] = pd.Series([jaccard(a, b) for a, b in zip(ml_g, tmdb_g)], index=df.index,
                                      dtype="float64")
    for g in CANON_GENRES:
        df[f"g_{slug(g)}"] = df["genres_unified"].map(lambda L, g=g: int(g in L)).astype("int8")

    # --- langue : catégories = langues les plus fréquentes du catalogue + other + und ------------
    df["original_language"] = df["original_language"].fillna("und")
    top = (df.loc[df["original_language"] != "und", "original_language"].value_counts()
             .head(params["preprocessing"]["meta"]["top_languages"]).index)
    df["lang_cat"] = df["original_language"].where(
        df["original_language"].isin(top) | (df["original_language"] == "und"), "other")

    # --- année : date de sortie TMDB, sinon année du titre MovieLens --------------------------------
    ry = df["release_date"].dt.year.astype("Int64")
    df["release_year"] = ry.where(ry.notna(), df["year_title"])

    # --- indicateurs et masques de modalités --------------------------------------------------------
    for c in ("has_overview", "has_poster", "in_tmdb", "in_kaggle", "overview_n_chars", "overview_n_words"):
        df[c] = df[c].fillna(0).astype("int8" if c.startswith(("has", "in_")) else "int32")
    df["year_missing"] = df["release_year"].isna().astype("int8")
    df["runtime_missing"] = df["runtime_min"].isna().astype("int8")
    df["snapshot_features_missing"] = df["popularity"].isna().astype("int8")
    df["has_meta"] = ((df["in_tmdb"] == 1) | (df["in_kaggle"] == 1) | (df["_ext_meta"] == 1)).astype("int8")
    df["n_modalities"] = (df["has_overview"] + df["has_poster"] + df["has_meta"]).astype("int8")
    df["all_modalities"] = (df["n_modalities"] == 3).astype("int8")
    df["modality_mask"] = [("T" if t else "-") + ("I" if i else "-") + ("M" if m else "-")
                           for t, i, m in zip(df["has_overview"], df["has_poster"], df["has_meta"])]
    out = df[MOVIE_COLUMNS].sort_values("movie_id").reset_index(drop=True)

    n = len(out)
    with_key = out["tmdb_id"].notna()
    info = {
        "movies": n,
        "join_rate_tmdb_key": float(with_key.mean()) if n else None,
        "join_rate_text": float(out["has_overview"].mean()) if n else None,
        "join_rate_image": float(out["has_poster"].mean()) if n else None,
        "join_rate_meta": float(out["has_meta"].mean()) if n else None,
        "modality_masks": out["modality_mask"].value_counts().to_dict(),
        "movies_by_source": out["source"].value_counts().to_dict(),
        "genre_mean_jaccard": float(out["genre_agreement"].mean()) if out["genre_agreement"].notna().any() else None,
    }
    return out, info


def stage(ctx: StageContext) -> None:
    i = ctx.interim
    out, info = align(ctx.params, pd.read_parquet(i / "catalog.parquet"), pd.read_parquet(i / "text.parquet"),
                      pd.read_parquet(i / "images.parquet"), pd.read_parquet(i / "meta.parquet"),
                      pd.read_parquet(i / "external" / "movies.parquet"))
    write_parquet(out, i / "movies_aligned.parquet")
    for k, v in info.items():
        ctx.metric(k, v)


if __name__ == "__main__":
    main_for("align", stage, "Apparie les modalités et marque les absences")
