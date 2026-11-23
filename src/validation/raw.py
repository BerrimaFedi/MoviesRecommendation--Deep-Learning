"""Étape `validate_raw` : contrôles sur les sources brutes, avant tout prétraitement.

- MovieLens : colonnes, types, domaines (note ∈ [0,5 ; 5], horodatage plausible), clés uniques,
  couverture des liens vers TMDB, volumes minimaux.
- TMDB : cache JSONL lisible, statuts connus, champs attendus présents.
- Affiches : chaque entrée « ok » du manifeste a son fichier, décodable, en RGB, de taille suffisante.
- Kaggle (facultatif) : colonnes attendues si la source est présente, sinon absence marquée.
Sortie : reports/validation/raw.json ; code de sortie 1 si un contrôle critique échoue.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common.config import Settings
from ..common.stage import StageContext, main_for
from ..ingestion import movielens
from ..ingestion.kaggle import UNAVAILABLE
from ..ingestion.posters import load_manifest, poster_file
from ..preprocessing.image import decodes
from .checks import Suite
from .schemas import KAGGLE_REQUIRED, RAW_ML_LINKS, RAW_ML_MOVIES, TMDB_REQUIRED_KEYS, check_schema, raw_ratings_spec


def validate_raw(params: dict) -> Suite:
    cfg = Settings.from_params(params)
    v = params["validation"]["raw"]
    suite = Suite("raw")

    # --- MovieLens -------------------------------------------------------------------------------
    movies, links, ratings = movielens.load(cfg.movielens_dir)
    check_schema(suite, movies, "movielens.movies", RAW_ML_MOVIES)
    check_schema(suite, links, "movielens.links", RAW_ML_LINKS)
    check_schema(suite, ratings, "movielens.ratings", raw_ratings_spec(params))
    suite.check("movielens: volume minimal de films", len(movies) >= v["min_movies"], category="source",
                observed=len(movies), threshold=v["min_movies"])
    suite.check("movielens: volume minimal de notes", len(ratings) >= v["min_ratings"], category="source",
                observed=len(ratings), threshold=v["min_ratings"])
    suite.rate_at_most("movielens: notes sur un film absent de movies.csv", ~ratings["movieId"].isin(movies["movieId"]),
                       max_rate=0.0, category="inter_modal")
    suite.check("movielens: links couvre tous les films", set(movies["movieId"]) == set(links["movieId"]),
                category="inter_modal", observed=len(set(movies["movieId"]) ^ set(links["movieId"])), threshold=0)
    suite.rate_at_most("movielens: films sans tmdbId", links["tmdbId"].isna(), max_rate=0.01,
                       category="missing", severity="warn")
    suite.rate_at_most("movielens: doublons (userId, movieId)", ratings.duplicated(["userId", "movieId"]),
                       max_rate=0.0, category="duplicates", severity="warn",
                       detail="corrigé au prétraitement (la note la plus récente est gardée)")

    # --- TMDB --------------------------------------------------------------------------------------
    lines = cfg.tmdb_cache.read_text(encoding="utf-8").splitlines() if cfg.tmdb_cache.exists() else []
    suite.check("tmdb: cache présent et non vide", bool(lines), category="source", observed=len(lines), threshold="> 0")
    parsed, bad = [], 0
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    suite.rate_at_most("tmdb: lignes JSONL illisibles", bad, len(lines), max_rate=v["max_tmdb_parse_error_rate"],
                       category="schema")
    statuses = pd.Series([r.get("status") for r in parsed], dtype="object")
    suite.rate_at_most("tmdb: statut ∈ {200, 404}", ~statuses.isin([200, 404]), max_rate=0.0, category="schema")
    ok = [r["data"] for r in parsed if r.get("status") == 200 and isinstance(r.get("data"), dict)]
    missing_keys = sum(1 for d in ok if any(k not in d for k in TMDB_REQUIRED_KEYS))
    suite.rate_at_most("tmdb: champs attendus présents", missing_keys, len(ok), max_rate=0.0, category="schema")
    wanted = {int(i) for i in links["tmdbId"].dropna()}
    got = {int(r["tmdb_id"]) for r in parsed if "tmdb_id" in r}
    suite.rate_at_most("tmdb: tmdbId du catalogue non encore collectés", len(wanted - got), len(wanted),
                       max_rate=0.01, category="missing", severity="warn")

    # --- affiches ----------------------------------------------------------------------------------
    man = load_manifest(cfg.posters_manifest)
    ok_ids = [t for t, r in man.items() if r.get("status") == "ok" and t in wanted]
    missing_files = [t for t in ok_ids if not poster_file(cfg, t).exists()]
    suite.rate_at_most("affiches: fichier présent pour chaque entrée ok", len(missing_files), len(ok_ids),
                       max_rate=v["max_poster_missing_file_rate"], category="inter_modal")
    ip = params["preprocessing"]["image"]
    present = [t for t in ok_ids if poster_file(cfg, t).exists()]
    if ip["verify_decode"] == "sample" and len(present) > ip["verify_sample"]:
        present = list(np.random.default_rng(params["project"]["seed"]).choice(present, ip["verify_sample"],
                                                                                 replace=False))
    if ip["verify_decode"] != "none":
        failed = [t for t in present if not decodes(poster_file(cfg, int(t)), ip["min_side"])]
        suite.rate_at_most("affiches: JPEG décodable, RGB, côté ≥ min_side", len(failed), len(present),
                           max_rate=v["max_poster_decode_error_rate"], category="schema",
                           detail=f"exemples : {failed[:5]}")
    outside = len(set(man) - wanted)
    suite.check("affiches: manifeste limité au catalogue", outside == 0, category="source", severity="warn",
                observed=outside, threshold=0,
                detail="entrées héritées de la Tâche 2 (secours Kaggle non filtré) : ignorées en aval")

    # --- Kaggle (facultatif) ------------------------------------------------------------------------------
    kfile = cfg.kaggle_dir / cfg.kaggle_file
    if kfile.exists():
        cols = pd.read_csv(kfile, nrows=0).columns
        absent = [c for c in KAGGLE_REQUIRED if c not in cols]
        suite.check("kaggle: colonnes attendues", not absent, category="schema", observed=absent or "toutes",
                    threshold=KAGGLE_REQUIRED)
    else:
        suite.check("kaggle: absence explicitement marquée", (cfg.kaggle_dir / UNAVAILABLE).exists(),
                    category="source", severity="warn", observed="absent", threshold=UNAVAILABLE)
    return suite


def stage(ctx: StageContext) -> None:
    suite = validate_raw(ctx.params)
    suite.write(ctx.reports / "validation" / "raw.json")
    suite.enforce(ctx)


if __name__ == "__main__":
    main_for("validate_raw", stage, "Contrôles de qualité des sources brutes")
