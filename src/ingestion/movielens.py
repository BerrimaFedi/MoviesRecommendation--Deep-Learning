"""Étape `ingest_movielens` : téléchargement (MD5 vérifié quand GroupLens le publie) et lecture.

Sortie : paths.raw_movielens/{movies,links,ratings,tags}.csv + SOURCE_MD5.txt (traçabilité).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from ..common.config import Settings
from ..common.stage import StageContext, main_for

CHUNK = 1 << 20


def _urlopen(url: str, timeout: int = 60):
    """GET avec le magasin de certificats certifi (utile sous Windows), puis celui du système."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=timeout, context=ctx)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_md5(url: str) -> str | None:
    """GroupLens publie un .md5 pour certaines archives (ex. ml-25m, ml-32m) mais pas toutes."""
    try:
        with _urlopen(url + ".md5", timeout=30) as r:
            text = r.read().decode("utf-8", "replace").strip()
        m = re.search(r"[0-9a-fA-F]{32}", text)
        return m.group(0).lower() if m else None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def download(cfg: Settings, force: bool = False, log=print) -> Path:
    """Télécharge et extrait la version MovieLens choisie dans cfg.movielens_dir. Idempotent."""
    target = cfg.movielens_dir
    if (target / "ratings.csv").exists() and not force:
        log(f"déjà présent : {target}")
        return target

    url = f"{cfg.movielens_base_url}/{cfg.movielens_version}.zip"
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{cfg.movielens_version}.zip"
        log(f"téléchargement {url}")
        with _urlopen(url, timeout=60) as r, zip_path.open("wb") as out:
            shutil.copyfileobj(r, out, CHUNK)
        expected, actual = _expected_md5(url), _md5(zip_path)
        if expected is not None and expected != actual:
            raise RuntimeError(f"MD5 invalide pour {zip_path.name} : attendu {expected}, obtenu {actual}")
        log("MD5 vérifié" if expected else f"pas de MD5 publié pour cette archive (MD5 local : {actual})")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        inner = Path(tmp) / cfg.movielens_version
        if not (inner / "ratings.csv").exists():
            raise RuntimeError(f"Archive extraite mais {inner.name}/ratings.csv est introuvable")
        target.mkdir(parents=True, exist_ok=True)
        for f in inner.iterdir():
            shutil.copy2(f, target / f.name)
    (target / "SOURCE_MD5.txt").write_text(actual + "\n", encoding="utf-8")
    return target


def load(movielens_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retourne (movies, links, ratings) avec des types stricts."""
    movies = pd.read_csv(movielens_dir / "movies.csv", dtype={"movieId": "int32"})
    links = pd.read_csv(movielens_dir / "links.csv",
                        dtype={"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"})
    ratings = pd.read_csv(movielens_dir / "ratings.csv",
                          dtype={"userId": "int32", "movieId": "int32",
                                 "rating": "float32", "timestamp": "int64"})
    return movies, links, ratings


def stage(ctx: StageContext) -> None:
    cfg = Settings.from_params(ctx.params)
    if ctx.params["ingestion"]["mode"] == "synthetic":
        from .synthetic import make_movielens
        make_movielens(ctx.params)
    else:
        download(cfg, log=ctx.info)
    movies, links, ratings = load(cfg.movielens_dir)
    ctx.metric("movies", len(movies))
    ctx.metric("links", len(links))
    ctx.metric("ratings", len(ratings))
    ctx.metric("users", ratings["userId"].nunique())
    md5 = cfg.movielens_dir / "SOURCE_MD5.txt"
    ctx.metric("source_md5", md5.read_text().strip() if md5.exists() else None)


if __name__ == "__main__":
    main_for("ingest_movielens", stage, "Télécharge et extrait MovieLens")
