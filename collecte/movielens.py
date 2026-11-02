"""Téléchargement (avec vérification MD5 quand GroupLens la publie) et lecture de MovieLens."""
from __future__ import annotations

import hashlib
import re
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import Settings

CHUNK = 1 << 20


def _urlopen(url: str, timeout: int = 60):
    """Effectue une requête GET avec gestion automatique des certificats SSL (notamment sous Windows)."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, timeout=timeout, context=ctx)
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context()
        return urllib.request.urlopen(url, timeout=timeout, context=ctx)
    except (ssl.SSLCertVerificationError, urllib.error.URLError, Exception):
        ctx = ssl._create_unverified_context()
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


def download(cfg: Settings, force: bool = False) -> Path:
    """Télécharge et extrait la version MovieLens choisie. Idempotent."""
    cfg.ensure_dirs()
    target = cfg.movielens_dir
    if (target / "ratings.csv").exists() and not force:
        print(f"[movielens] déjà présent : {target}")
        return target

    url = f"{cfg.movielens_base_url}/{cfg.movielens_version}.zip"
    zip_path = cfg.raw_dir / f"{cfg.movielens_version}.zip"
    print(f"[movielens] téléchargement {url}")
    with _urlopen(url, timeout=60) as r, zip_path.open("wb") as out:
        shutil.copyfileobj(r, out, CHUNK)

    expected = _expected_md5(url)
    actual = _md5(zip_path)
    if expected is None:
        print(f"[movielens] pas de somme MD5 publiée pour cette archive (MD5 local : {actual})")
    elif expected != actual:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 invalide pour {zip_path.name} : attendu {expected}, obtenu {actual}")
    else:
        print("[movielens] MD5 vérifié")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cfg.raw_dir)
    if not (target / "ratings.csv").exists():
        raise RuntimeError(f"Archive extraite mais {target}/ratings.csv est introuvable")
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
