"""Sources synthétiques au format exact des vraies sources (mode `ingestion.mode: synthetic`).

Usage : intégration continue et démonstration hors ligne. MovieLens et TMDB interdisent la
redistribution : on ne peut pas versionner un extrait réel dans le dépôt public, donc la CI exécute le
pipeline complet sur ce petit jeu artificiel, généré de façon déterministe (graine project.seed).

Le jeu contient volontairement quelques défauts réalistes (tmdbId absent, film inconnu de TMDB,
synopsis vide, affiche manquante) pour que les contrôles de qualité aient quelque chose à mesurer.
Les genres influencent la couleur de l'affiche, le vocabulaire du synopsis et les notes : le signal
multimodal est donc réel (utile au prototype de fusion).
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..common.config import Settings

GENRES = [("Action", "Action", (200, 40, 30)), ("Comedy", "Comedy", (240, 200, 40)),
          ("Drama", "Drama", (60, 60, 140)), ("Horror", "Horror", (20, 20, 20)),
          ("Romance", "Romance", (230, 100, 160)), ("Sci-Fi", "Science Fiction", (30, 160, 200)),
          ("Animation", "Animation", (120, 220, 90)), ("Thriller", "Thriller", (90, 40, 90))]
WORDS = {
    "Action": "explosion chase mission agent fight rescue",
    "Comedy": "funny wedding friends mishap prank laugh",
    "Drama": "family loss memory struggle father truth",
    "Horror": "haunted night curse blood scream demon",
    "Romance": "love heart summer letter kiss promise",
    "Sci-Fi": "space robot future planet alien ship",
    "Animation": "magic kingdom talking animals adventure toy",
    "Thriller": "murder detective secret conspiracy trap witness",
}
FETCHED_AT = "2026-01-01T00:00:00+00:00"
BASE_TS = 1420070400  # 2015-01-01


SYNTHETIC_TAG = "SYNTHÉTIQUE"


class RefuseOverwrite(RuntimeError):
    pass


def guard(params: dict[str, Any]) -> None:
    """Refuse d'écrire des données synthétiques par-dessus de vraies données collectées."""
    cfg = Settings.from_params(params)
    readme = cfg.movielens_dir / "README.txt"
    if (cfg.movielens_dir / "ratings.csv").exists() and (
            not readme.exists() or SYNTHETIC_TAG not in readme.read_text(encoding="utf-8", errors="replace")):
        raise RefuseOverwrite(f"{cfg.movielens_dir} contient de vraies données MovieLens : utilisez des chemins "
                              "dédiés pour le mode synthétique (voir configs/params.ci.yaml)")
    if cfg.tmdb_cache.exists():
        with cfg.tmdb_cache.open(encoding="utf-8") as f:
            first = f.readline()
        if first and FETCHED_AT not in first:
            raise RefuseOverwrite(f"{cfg.tmdb_cache} contient un vrai cache TMDB : refus de l'écraser")


def _catalog(params: dict[str, Any]) -> pd.DataFrame:
    s = params["ingestion"]["synthetic"]
    rng = np.random.default_rng(params["project"]["seed"])
    rows = []
    for mid in range(1, int(s["n_movies"]) + 1):
        k = int(rng.integers(1, 4))
        gidx = sorted(rng.choice(len(GENRES), size=k, replace=False).tolist())
        rows.append({"movieId": mid, "tmdbId": 1000 + mid, "imdbId": 100000 + mid,
                     "year": int(rng.integers(1975, 2012)), "gidx": gidx,
                     "runtime": int(rng.integers(80, 160)), "popularity": float(rng.gamma(2.0, 3.0))})
    df = pd.DataFrame(rows)
    df.loc[df["movieId"] == 2, "tmdbId"] = pd.NA      # tmdbId absent de links.csv
    return df


def _special(params: dict[str, Any]) -> dict[str, int]:
    """tmdbId des cas particuliers (positions fixes dans le catalogue)."""
    return {"tmdb_404": 1003, "empty_overview": 1004, "no_poster": 1005}


def make_movielens(params: dict[str, Any]) -> None:
    guard(params)
    cfg = Settings.from_params(params)
    d = cfg.movielens_dir
    d.mkdir(parents=True, exist_ok=True)
    cat = _catalog(params)
    s = params["ingestion"]["synthetic"]
    rng = np.random.default_rng(params["project"]["seed"] + 1)
    movies = pd.DataFrame({
        "movieId": cat["movieId"],
        "title": [f"Synthetic Movie {m} ({y})" for m, y in zip(cat["movieId"], cat["year"])],
        "genres": ["|".join(GENRES[i][0] for i in g) for g in cat["gidx"]],
    })
    links = pd.DataFrame({"movieId": cat["movieId"], "imdbId": cat["imdbId"],
                          "tmdbId": cat["tmdbId"].astype("Int64")})
    # utilisateurs : préférence par genre -> note ; popularité inégale des films (longue traîne)
    pop = np.asarray(cat["popularity"], dtype=float)
    p_item = pop / pop.sum()
    ratings = []
    for u in range(1, int(s["n_users"]) + 1):
        taste = rng.normal(0, 1, len(GENRES))
        n = min(int(s["ratings_per_user"]), len(cat))
        items = rng.choice(len(cat), size=n, replace=False, p=p_item)
        for j in items:
            score = 3.2 + 0.6 * np.mean([taste[g] for g in cat.loc[j, "gidx"]]) + rng.normal(0, 0.5)
            rating = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))
            ratings.append((u, int(cat.loc[j, "movieId"]), rating,
                            BASE_TS + int(rng.integers(0, 3 * 365 * 86400))))
    r = pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"])
    movies.to_csv(d / "movies.csv", index=False)
    links.to_csv(d / "links.csv", index=False)
    r.to_csv(d / "ratings.csv", index=False)
    pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"]).to_csv(d / "tags.csv", index=False)
    md5 = hashlib.md5((d / "ratings.csv").read_bytes()).hexdigest()
    (d / "SOURCE_MD5.txt").write_text(f"{md5}\n", encoding="utf-8")
    (d / "README.txt").write_text(f"Jeu MovieLens {SYNTHETIC_TAG} généré par src/ingestion/synthetic.py\n",
                                  encoding="utf-8")


def _overview(rng: np.random.Generator, gidx: list[int], tid: int) -> str:
    words = " ".join(WORDS[GENRES[i][0]] for i in gidx).split()
    picked = rng.choice(words, size=min(len(words), 12), replace=False)
    return f"In this story number {tid}, " + " ".join(picked) + " lead to an unexpected ending."


def make_tmdb(params: dict[str, Any]) -> None:
    guard(params)
    cfg = Settings.from_params(params)
    cfg.tmdb_dir.mkdir(parents=True, exist_ok=True)
    cat = _catalog(params)
    sp = _special(params)
    rng = np.random.default_rng(params["project"]["seed"] + 2)
    with cfg.tmdb_cache.open("w", encoding="utf-8") as f:
        for row in cat.itertuples():
            if pd.isna(row.tmdbId):
                continue
            tid = int(row.tmdbId)
            if tid == sp["tmdb_404"]:
                rec = {"tmdb_id": tid, "status": 404, "fetched_at": FETCHED_AT, "data": None}
            else:
                data = {
                    "id": tid, "title": f"Synthetic Movie {row.movieId}",
                    "overview": "" if tid == sp["empty_overview"] else _overview(rng, row.gidx, tid),
                    "tagline": "", "runtime": row.runtime, "popularity": round(row.popularity, 3),
                    "vote_average": round(float(rng.uniform(4, 8.5)), 1), "vote_count": int(rng.integers(5, 5000)),
                    "original_language": "en" if rng.random() < 0.85 else str(rng.choice(["fr", "ja", "it"])),
                    "release_date": f"{row.year}-0{int(rng.integers(1, 10))}-15",
                    "genres": [{"id": i, "name": GENRES[i][1]} for i in row.gidx],
                    "poster_path": None if tid == sp["no_poster"] else f"/syn{tid}.jpg", "adult": False,
                }
                rec = {"tmdb_id": tid, "status": 200, "fetched_at": FETCHED_AT, "data": data}
            f.write(json.dumps(rec) + "\n")
    cfg.tmdb_config_file.write_text(json.dumps({"images": {"secure_base_url": "synthetic://",
                                                           "poster_sizes": [cfg.poster_size]}}), encoding="utf-8")


def make_posters(params: dict[str, Any]) -> None:
    guard(params)
    from .tmdb import cache_to_records

    cfg = Settings.from_params(params)
    cfg.posters_dir.mkdir(parents=True, exist_ok=True)
    cat = _catalog(params).dropna(subset=["tmdbId"]).set_index("tmdbId")
    rng = np.random.default_rng(params["project"]["seed"] + 3)
    with cfg.posters_manifest.open("w", encoding="utf-8") as man:
        for rec in cache_to_records(cfg.tmdb_cache):
            tid, path = rec["_tmdb_id"], rec.get("poster_path")
            if not path:
                continue
            base = np.array(GENRES[cat.loc[tid, "gidx"][0]][2], dtype=float)
            arr = np.clip(base + rng.normal(0, 25, (513, 342, 3)), 0, 255).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
            data = buf.getvalue()
            Path(cfg.posters_dir / f"{tid}.jpg").write_bytes(data)
            man.write(json.dumps({"tmdb_id": tid, "poster_path": path, "fetched_at": FETCHED_AT, "status": "ok",
                                  "width": 342, "height": 513, "bytes": len(data),
                                  "sha1": hashlib.sha1(data).hexdigest()}) + "\n")


def make_kaggle(params: dict[str, Any]) -> None:
    """Secours : un synopsis pour le film dont le synopsis TMDB est vide."""
    guard(params)
    cfg = Settings.from_params(params)
    cfg.kaggle_dir.mkdir(parents=True, exist_ok=True)
    tid = _special(params)["empty_overview"]
    cols = ["id", "imdb_id", "title", "overview", "tagline", "genres", "runtime", "popularity",
            "vote_average", "vote_count", "original_language", "release_date", "poster_path"]
    rows = [[str(tid), "tt0000000", "Synthetic", "Fallback synopsis from the synthetic Kaggle snapshot, long "
             "enough to be kept.", "", "[{'id': 18, 'name': 'Drama'}]", "100", "1", "6", "10", "en",
             "2000-01-01", f"/syn{tid}.jpg"],
            ["1997-08-20", "", "Shifted row", "", "", "", "", "", "", "", "", "", ""]]
    pd.DataFrame(rows, columns=cols).to_csv(cfg.kaggle_dir / cfg.kaggle_file, index=False)
