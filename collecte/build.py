"""Construction du jeu final : fusion des sources, normalisation, découpages, rapport qualité.

Sorties dans data/processed/ :
  movies.parquet (+ movies.csv)  une ligne par film : identifiants, texte, chemin d'affiche,
                                 métadonnées normalisées, indicateurs de modalités manquantes
  ratings.parquet                notes + colonne `split` (train / val / test / cold_item_test)
  manifest.json                  traçabilité (versions, empreintes, comptes, date de collecte)
Sorties dans data/reports/ : quality_report.json et quality_report.md
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import kaggle_meta, movielens, posters, tmdb
from .config import Settings
from .normalize import (CANON_GENRES, clean_text, genre_columns, imdb_tt, jaccard, normalize_language,
                        parse_ml_genres, parse_tmdb_genres, slug, split_title_year, to_float)

TEXT_MIN_LEN = 20  # un synopsis plus court est considéré comme absent (souvent un titre ou « TBD »)
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."

TMDB_COLS = ["tmdb_id", "tmdb_fetched_at", "t_title", "t_overview", "t_tagline", "genres_tmdb",
             "t_runtime", "t_popularity", "t_vote_average", "t_vote_count", "t_language",
             "t_release_date", "t_poster_path", "t_adult"]


def _py(x: Any) -> Any:
    """Convertit numpy/pandas en types JSON natifs."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return x


def _coalesce(primary: pd.Series, fallback: pd.Series, names: tuple[str, str] = ("tmdb", "kaggle")):
    value = primary.where(primary.notna(), fallback)
    src = pd.Series(np.where(primary.notna(), names[0], np.where(fallback.notna(), names[1], None)),
                    index=primary.index, dtype=object)
    return value, src


def _tmdb_frame(cfg: Settings, unmapped: dict[str, int]) -> tuple[pd.DataFrame, int]:
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


def _kaggle_frame(cfg: Settings, enabled: bool) -> pd.DataFrame:
    cols = ["tmdb_id", "k_overview", "k_tagline", "genres_kaggle", "k_runtime", "k_language",
            "k_release_date", "k_poster_path"]
    k = kaggle_meta.load(cfg) if enabled else None
    if k is None:
        out = pd.DataFrame(columns=cols)
    else:
        out = pd.DataFrame({
            "tmdb_id": k["tmdb_id"], "k_overview": k["overview"], "k_tagline": k["tagline"],
            "genres_kaggle": k["genres"], "k_runtime": k["runtime"], "k_language": k["original_language"],
            "k_release_date": k["release_date"], "k_poster_path": k["poster_path"],
        })
    out["tmdb_id"] = out["tmdb_id"].astype("Int64")
    return out


def build_movies(cfg: Settings, use_kaggle: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    ml_movies, ml_links, _ = movielens.load(cfg.movielens_dir)
    info: dict[str, Any] = {}
    unmapped_ml: dict[str, int] = {}
    unmapped_tmdb: dict[str, int] = {}

    base = ml_movies.merge(ml_links, on="movieId", how="left")
    parsed = base["genres"].map(lambda s: parse_ml_genres(s, unmapped_ml))
    base["genres_ml"] = parsed.map(lambda t: t[0])
    base["is_imax"] = parsed.map(lambda t: int(t[1])).astype("int8")
    ty = base["title"].map(split_title_year)
    base["title_clean"] = ty.map(lambda t: t[0])
    base["year_title"] = pd.array(ty.map(lambda t: t[1]).tolist(), dtype="Int64")
    base["imdb_tt"] = base["imdbId"].map(imdb_tt)
    base = base.rename(columns={"tmdbId": "tmdb_id"}).drop(columns=["genres", "imdbId"])
    base["tmdb_id"] = base["tmdb_id"].astype("Int64")

    t, n404 = _tmdb_frame(cfg, unmapped_tmdb)
    k = _kaggle_frame(cfg, use_kaggle)
    df = base.merge(t, on="tmdb_id", how="left").merge(k, on="tmdb_id", how="left")
    df["in_tmdb"] = df["tmdb_fetched_at"].notna().astype("int8")
    df["in_kaggle"] = df["k_language"].notna().astype("int8") | df["k_overview"].notna().astype("int8")
    df["in_kaggle"] = df["in_kaggle"].astype("int8")

    # --- texte -------------------------------------------------------------------------
    df["overview"], df["overview_source"] = _coalesce(df["t_overview"], df["k_overview"])
    df["tagline"], _ = _coalesce(df["t_tagline"], df["k_tagline"])
    df["has_overview"] = (df["overview"].fillna("").str.len() >= TEXT_MIN_LEN).astype("int8")

    # --- genres : union MovieLens + TMDB(/Kaggle), vocabulaire canonique ------------------
    def _tmdb_like(row_t: Any, row_k: Any) -> list[str]:
        if isinstance(row_t, list) and row_t:
            return row_t
        return row_k if isinstance(row_k, list) else []

    tmdb_like = [_tmdb_like(a, b) for a, b in zip(df["genres_tmdb"], df["genres_kaggle"])]
    order = {g: i for i, g in enumerate(CANON_GENRES)}
    df["genres_unified"] = [sorted(set(a) | set(b), key=order.__getitem__)
                            for a, b in zip(df["genres_ml"], tmdb_like)]
    jac = [jaccard(a, b) for a, b in zip(df["genres_ml"], tmdb_like)]
    df["genre_agreement"] = pd.Series(jac, index=df.index, dtype="float64")
    for g in CANON_GENRES:
        df[f"g_{slug(g)}"] = df["genres_unified"].map(lambda L, g=g: int(g in L)).astype("int8")

    # --- langue ---------------------------------------------------------------------------
    lang_t = df["t_language"].where(df["t_language"].fillna("").astype(str).str.strip() != "")
    lang, _ = _coalesce(lang_t, df["k_language"])
    df["original_language"] = lang.map(normalize_language)
    top = (df.loc[df["original_language"] != "und", "original_language"].value_counts()
             .head(cfg.top_languages).index)
    df["lang_cat"] = df["original_language"].where(
        df["original_language"].isin(top) | (df["original_language"] == "und"), "other")

    # --- dates ----------------------------------------------------------------------------
    rd, _ = _coalesce(df["t_release_date"].replace("", np.nan), df["k_release_date"].replace("", np.nan))
    df["release_date"] = pd.to_datetime(rd, errors="coerce")
    ry = df["release_date"].dt.year.astype("Int64")
    df["release_year"] = ry.where(ry.notna(), df["year_title"])
    df["year_missing"] = df["release_year"].isna().astype("int8")

    # --- numériques ------------------------------------------------------------------------
    runtime, _ = _coalesce(df["t_runtime"], df["k_runtime"])
    runtime = pd.to_numeric(runtime, errors="coerce")
    df["runtime_min"] = runtime.where(runtime > 0)
    df["runtime_missing"] = df["runtime_min"].isna().astype("int8")
    # popularité et votes : TMDB uniquement (instantané daté `tmdb_fetched_at`). On ne mélange pas
    # avec l'instantané Kaggle de 2017 : les échelles et les dates ne sont pas comparables.
    df["popularity"] = pd.to_numeric(df["t_popularity"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["t_vote_count"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["t_vote_average"], errors="coerce").where(df["vote_count"] > 0)
    df["log_popularity"] = np.log1p(df["popularity"])
    df["log_vote_count"] = np.log1p(df["vote_count"])
    df["snapshot_features_missing"] = df["popularity"].isna().astype("int8")

    # --- affiches ---------------------------------------------------------------------------
    df["poster_path"], df["poster_path_source"] = _coalesce(df["t_poster_path"], df["k_poster_path"])
    manifest = posters.load_manifest(cfg.posters_manifest)
    pm = pd.DataFrame(list(manifest.values()),
                      columns=["tmdb_id", "status", "width", "height", "sha1"])
    pm = pm[pm["status"] == "ok"].rename(columns={"width": "poster_w", "height": "poster_h",
                                                    "sha1": "poster_sha1"}).drop(columns="status")
    pm["tmdb_id"] = pm["tmdb_id"].astype("Int64")
    df = df.merge(pm, on="tmdb_id", how="left")
    df["has_poster"] = df["poster_sha1"].notna().astype("int8")
    df["poster_file"] = [f"posters/{int(i)}.jpg" if h else None
                         for i, h in zip(df["tmdb_id"].fillna(0), df["has_poster"])]

    # --- modalités ----------------------------------------------------------------------------
    df["has_meta"] = ((df["in_tmdb"] == 1) | (df["in_kaggle"] == 1)).astype("int8")
    df["n_modalities"] = (df["has_overview"] + df["has_poster"] + df["has_meta"]).astype("int8")
    df["all_modalities"] = (df["n_modalities"] == 3).astype("int8")

    df = df.rename(columns={"movieId": "movie_id", "title_clean": "title", "title": "title_movielens",
                            "t_title": "title_tmdb"})
    cols = (["movie_id", "tmdb_id", "imdb_tt", "title", "title_movielens", "title_tmdb", "overview",
             "overview_source", "tagline", "poster_path", "poster_path_source", "poster_file",
             "poster_w", "poster_h", "poster_sha1", "release_date", "release_year", "year_title",
             "original_language", "lang_cat", "runtime_min", "popularity", "vote_average", "vote_count",
             "log_popularity", "log_vote_count", "genres_ml", "genres_unified", "genre_agreement", "is_imax"]
            + genre_columns()
            + ["has_overview", "has_poster", "has_meta", "in_tmdb", "in_kaggle", "runtime_missing",
               "year_missing", "snapshot_features_missing", "n_modalities", "all_modalities",
               "tmdb_fetched_at"])
    out = df[cols].sort_values("movie_id").reset_index(drop=True)

    info.update(n404=n404, unmapped_ml=unmapped_ml, unmapped_tmdb=unmapped_tmdb,
                tmdb_id_missing_in_links=int(base["tmdb_id"].isna().sum()),
                tmdb_id_duplicated_in_links=int(base["tmdb_id"].dropna().duplicated().sum()))
    return out, info


def build_ratings(cfg: Settings, movies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, _, r = movielens.load(cfg.movielens_dir)
    n0 = len(r)
    r = r.drop_duplicates(["userId", "movieId"], keep="last")
    r = r[r["movieId"].isin(movies["movie_id"]) & r["rating"].between(0.5, 5.0)].copy()
    r = r.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    info: dict[str, Any] = {"ratings_raw": n0, "ratings_kept": len(r)}

    # Films « froids » : retirés de l'entraînement, évalués à part (démarrage à froid strict).
    counts = r.groupby("movie_id").size()
    has_content = movies.set_index("movie_id")[["has_overview", "has_poster"]].max(axis=1) > 0
    eligible = sorted(set(counts[counts >= cfg.cold_item_min_ratings].index)
                      & set(has_content[has_content].index))
    rng = np.random.default_rng(cfg.seed)
    n_cold = int(round(len(eligible) * cfg.cold_item_frac))
    cold_ids = set(rng.choice(eligible, size=n_cold, replace=False).tolist()) if n_cold else set()

    r["split"] = "cold_item_test"
    warm_mask = ~r["movie_id"].isin(cold_ids)
    warm = r[warm_mask].sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
    n = len(warm)
    if n:
        t_train = warm["timestamp"].iloc[max(int(n * cfg.train_frac) - 1, 0)]
        t_val = warm["timestamp"].iloc[max(int(n * (cfg.train_frac + cfg.val_frac)) - 1, 0)]
        ts = r["timestamp"]
        r.loc[warm_mask & (ts <= t_train), "split"] = "train"
        r.loc[warm_mask & (ts > t_train) & (ts <= t_val), "split"] = "val"
        r.loc[warm_mask & (ts > t_val), "split"] = "test"
        info["train_end"] = datetime.fromtimestamp(int(t_train), timezone.utc).isoformat()
        info["val_end"] = datetime.fromtimestamp(int(t_val), timezone.utc).isoformat()

    train = r[r["split"] == "train"]
    r["user_in_train"] = r["user_id"].isin(set(train["user_id"]))
    r["item_in_train"] = r["movie_id"].isin(set(train["movie_id"]))
    r = r.sort_values(["timestamp", "user_id", "movie_id"], kind="stable").reset_index(drop=True)

    info["split_sizes"] = r["split"].value_counts().to_dict()
    info["cold_items"] = len(cold_ids)
    info["unseen_item_share_val_test"] = float(
        1 - r.loc[r["split"].isin(["val", "test"]), "item_in_train"].mean()) if (
        r["split"].isin(["val", "test"]).any()) else None
    return r, info


def quality_report(cfg: Settings, movies: pd.DataFrame, ratings: pd.DataFrame,
                   minfo: dict[str, Any], rinfo: dict[str, Any]) -> dict[str, Any]:
    n = len(movies)
    pm = posters.load_manifest(cfg.posters_manifest)
    status_counts = pd.Series([v["status"] for v in pm.values()], dtype="object").value_counts().to_dict()
    dup_sha = (movies.dropna(subset=["poster_sha1"]).groupby("poster_sha1")["tmdb_id"]
               .nunique().loc[lambda s: s > 1])
    cmp = movies["genre_agreement"].dropna()
    ymis = (movies["year_title"].notna() & movies["release_year"].notna()
            & ((movies["year_title"].astype("float") - movies["release_year"].astype("float")).abs() > 1))
    fetched = pd.to_datetime(movies["tmdb_fetched_at"], errors="coerce", utc=True).dropna()
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "movielens_version": cfg.movielens_version,
        "counts": {"movies": n, "ratings": len(ratings), "users": int(ratings["user_id"].nunique()),
                   "tmdb_id_missing_in_links": minfo["tmdb_id_missing_in_links"],
                   "tmdb_id_duplicated_in_links": minfo["tmdb_id_duplicated_in_links"],
                   "found_in_tmdb": int(movies["in_tmdb"].sum()), "tmdb_404": minfo["n404"],
                   "found_in_kaggle": int(movies["in_kaggle"].sum())},
        "coverage": {
            "has_overview": float(movies["has_overview"].mean()) if n else None,
            "has_poster": float(movies["has_poster"].mean()) if n else None,
            "has_meta": float(movies["has_meta"].mean()) if n else None,
            "all_three_modalities": float(movies["all_modalities"].mean()) if n else None,
            "neither_overview_nor_poster": int(((movies["has_overview"] == 0) & (movies["has_poster"] == 0)).sum()),
            "runtime_missing": float(movies["runtime_missing"].mean()) if n else None,
            "year_missing": float(movies["year_missing"].mean()) if n else None,
        },
        "sources": {
            "overview": movies["overview_source"].value_counts(dropna=False).to_dict(),
            "poster_path": movies["poster_path_source"].value_counts(dropna=False).to_dict(),
        },
        "genres": {
            "movies_compared_ml_vs_tmdb": int(len(cmp)),
            "mean_jaccard": float(cmp.mean()) if len(cmp) else None,
            "identical_share": float((cmp == 1.0).mean()) if len(cmp) else None,
            "movies_without_any_genre": int((movies["genres_unified"].map(len) == 0).sum()),
            "unmapped_movielens": minfo["unmapped_ml"], "unmapped_tmdb": minfo["unmapped_tmdb"],
        },
        "language": {"top": movies["original_language"].value_counts().head(10).to_dict(),
                     "undetermined": int((movies["original_language"] == "und").sum())},
        "years": {"title_vs_release_gap_gt_1y": int(ymis.sum())},
        "posters": {"manifest_status": status_counts, "shared_by_several_movies": int(len(dup_sha))},
        "splits": {k: rinfo[k] for k in rinfo if k not in ("ratings_raw", "ratings_kept")} | {
            "ratings_raw": rinfo["ratings_raw"], "ratings_kept": rinfo["ratings_kept"]},
        "tmdb_snapshot": {
            "oldest_fetch": fetched.min().isoformat() if len(fetched) else None,
            "newest_fetch": fetched.max().isoformat() if len(fetched) else None,
            "max_cache_age_days_allowed": cfg.cache_max_age_days,
        },
        "notes": [
            "popularity, vote_average et vote_count sont des instantanés TMDB à la date tmdb_fetched_at : "
            "ils reflètent l'attention actuelle, pas celle de la période des notes (risque de fuite temporelle "
            "dans un découpage chronologique). À traiter comme variables optionnelles ou à exclure en test.",
            "Le dataset Kaggle est un instantané TMDB de 2017 dont les notes viennent de MovieLens : "
            "ce n'est pas une source indépendante.",
            ATTRIBUTION,
        ],
    }
    return _py(rep)


def _md(rep: dict[str, Any]) -> str:
    c, cov, g, s = rep["counts"], rep["coverage"], rep["genres"], rep["splits"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f} %"  # noqa: E731
    lines = [
        "# Rapport qualité des données", "",
        f"Généré le {rep['generated_at']} · MovieLens `{rep['movielens_version']}`", "",
        "## Volumes", "",
        f"- Films : {c['movies']} · notes : {c['ratings']} · utilisateurs : {c['users']}",
        f"- Films sans tmdbId dans links.csv : {c['tmdb_id_missing_in_links']}",
        f"- Trouvés dans TMDB : {c['found_in_tmdb']} (404 : {c['tmdb_404']}) · dans Kaggle : {c['found_in_kaggle']}", "",
        "## Couverture des modalités", "",
        f"- Synopsis exploitable : {pct(cov['has_overview'])}",
        f"- Affiche valide : {pct(cov['has_poster'])}",
        f"- Métadonnées : {pct(cov['has_meta'])}",
        f"- Les trois modalités : {pct(cov['all_three_modalities'])}",
        f"- Ni synopsis ni affiche : {cov['neither_overview_nor_poster']} films",
        f"- Durée manquante : {pct(cov['runtime_missing'])} · année manquante : {pct(cov['year_missing'])}", "",
        "## Cohérence des genres (MovieLens vs TMDB/Kaggle)", "",
        f"- Films comparés : {g['movies_compared_ml_vs_tmdb']} · Jaccard moyen : "
        f"{'n/a' if g['mean_jaccard'] is None else round(g['mean_jaccard'], 3)} · "
        f"identiques : {pct(g['identical_share'])}",
        f"- Films sans aucun genre : {g['movies_without_any_genre']}",
        f"- Genres hors vocabulaire (MovieLens) : {g['unmapped_movielens'] or 'aucun'}",
        f"- Genres hors vocabulaire (TMDB) : {g['unmapped_tmdb'] or 'aucun'}", "",
        "## Découpages", "",
        f"- Tailles : {s.get('split_sizes')}",
        f"- Films retirés pour le démarrage à froid strict : {s.get('cold_items')}",
        f"- Part des notes val/test sur un film absent du train : {pct(s.get('unseen_item_share_val_test'))}", "",
        "## Avertissements", "",
    ] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(lines) + "\n"


def build(cfg: Settings, use_kaggle: bool = True, also_csv: bool = True) -> dict[str, Any]:
    cfg.ensure_dirs()
    movies, minfo = build_movies(cfg, use_kaggle)
    ratings, rinfo = build_ratings(cfg, movies)

    movies.to_parquet(cfg.processed_dir / "movies.parquet", index=False)
    ratings.to_parquet(cfg.processed_dir / "ratings.parquet", index=False)
    if also_csv:
        flat = movies.copy()
        for col in ("genres_ml", "genres_unified"):
            flat[col] = flat[col].map("|".join)
        flat.to_csv(cfg.processed_dir / "movies.csv", index=False)

    rep = quality_report(cfg, movies, ratings, minfo, rinfo)
    (cfg.reports_dir / "quality_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    (cfg.reports_dir / "quality_report.md").write_text(_md(rep), encoding="utf-8")

    md5_file = cfg.movielens_dir / "SOURCE_MD5.txt"
    manifest = {
        "built_at": rep["generated_at"], "movielens_version": cfg.movielens_version,
        "movielens_zip_md5": md5_file.read_text().strip() if md5_file.exists() else None,
        "tmdb_language": cfg.language, "poster_size": cfg.poster_size, "seed": cfg.seed,
        "split": {"train_frac": cfg.train_frac, "val_frac": cfg.val_frac,
                  "cold_item_frac": cfg.cold_item_frac, "cold_item_min_ratings": cfg.cold_item_min_ratings},
        "counts": rep["counts"], "movies_columns": list(movies.columns),
        "python": platform.python_version(), "pandas": pd.__version__, "attribution": ATTRIBUTION,
    }
    (cfg.processed_dir / "manifest.json").write_text(json.dumps(_py(manifest), indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(f"[build] {len(movies)} films, {len(ratings)} notes -> {cfg.processed_dir}")
    print(f"[build] rapport : {cfg.reports_dir / 'quality_report.md'}")
    return rep
