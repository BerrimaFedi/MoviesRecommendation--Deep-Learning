"""Point d'entrée : python -m collecte <commande>."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import build as build_mod
from . import kaggle_meta, movielens, posters, tmdb
from .config import MissingCredentials, Settings, load_dotenv


def _settings(a: argparse.Namespace) -> Settings:
    cfg = Settings(data_dir=Path(a.data_dir), movielens_version=a.movielens)
    # surcharges facultatives (serveur miroir, tests) ; par défaut les adresses officielles
    cfg.tmdb_base_url = os.environ.get("TMDB_BASE_URL", cfg.tmdb_base_url)
    cfg.tmdb_image_base = os.environ.get("TMDB_IMAGE_BASE", cfg.tmdb_image_base)
    if getattr(a, "rate", None):
        cfg.rate_per_sec = a.rate
    if getattr(a, "concurrency", None):
        cfg.concurrency = a.concurrency
    if getattr(a, "language", None):
        cfg.language = a.language
    if getattr(a, "poster_size", None):
        cfg.poster_size = a.poster_size
    return cfg


def _tmdb_ids(cfg: Settings, limit: int | None) -> list[int]:
    _, links, _ = movielens.load(cfg.movielens_dir)
    ids = sorted(int(i) for i in links["tmdbId"].dropna().unique())
    return ids[:limit] if limit else ids


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="collecte", description="Projet 10 — collecte et préparation des données")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--movielens", default="ml-latest-small",
                   help="ml-latest-small (dév), ml-25m ou ml-32m (résultats finaux)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-movielens", help="télécharge et extrait MovieLens")
    sub.add_parser("download-kaggle", help="télécharge The Movies Dataset (CLI Kaggle requise)")

    f = sub.add_parser("fetch-tmdb", help="interroge l'API TMDB pour chaque tmdbId")
    f.add_argument("--limit", type=int, help="ne traiter que les N premiers films (essais)")
    f.add_argument("--retry-404", action="store_true")
    f.add_argument("--rate", type=float, help="requêtes/seconde (défaut 20)")
    f.add_argument("--concurrency", type=int)
    f.add_argument("--language", help="langue des synopsis (défaut en-US)")
    f.add_argument("--poster-size", dest="poster_size")

    q = sub.add_parser("fetch-posters", help="télécharge et valide les affiches")
    q.add_argument("--retry-failed", action="store_true")
    q.add_argument("--rate", type=float)
    q.add_argument("--concurrency", type=int)
    q.add_argument("--poster-size", dest="poster_size")

    b = sub.add_parser("build", help="fusionne, normalise, découpe et produit le rapport qualité")
    b.add_argument("--no-kaggle", action="store_true")
    b.add_argument("--no-csv", action="store_true")

    r = sub.add_parser("run-all", help="enchaîne toutes les étapes")
    r.add_argument("--limit", type=int)
    r.add_argument("--no-kaggle", action="store_true")

    a = p.parse_args(argv)
    cfg = _settings(a)

    try:
        if a.cmd == "download-movielens":
            movielens.download(cfg)
        elif a.cmd == "download-kaggle":
            kaggle_meta.download(cfg)
        elif a.cmd == "fetch-tmdb":
            tmdb.fetch(cfg, _tmdb_ids(cfg, a.limit), retry_404=a.retry_404)
        elif a.cmd == "fetch-posters":
            posters.fetch(cfg, retry_failed=a.retry_failed)
        elif a.cmd == "build":
            build_mod.build(cfg, use_kaggle=not a.no_kaggle, also_csv=not a.no_csv)
        elif a.cmd == "run-all":
            movielens.download(cfg)
            if not a.no_kaggle:
                kaggle_meta.download(cfg)
            tmdb.fetch(cfg, _tmdb_ids(cfg, a.limit))
            posters.fetch(cfg)
            build_mod.build(cfg, use_kaggle=not a.no_kaggle)
    except MissingCredentials as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 2
    except tmdb.AuthError as e:
        print(f"Erreur : {e} Vérifiez TMDB_READ_TOKEN / TMDB_API_KEY.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
