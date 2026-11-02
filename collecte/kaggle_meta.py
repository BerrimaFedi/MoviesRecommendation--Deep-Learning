"""Dataset Kaggle « The Movies Dataset » : source de secours hors ligne et point de comparaison.

À savoir (vérifié) : ses notes proviennent de MovieLens et ses métadonnées sont un instantané TMDB
(films sortis jusqu'en juillet 2017). Ce n'est donc PAS une source indépendante de TMDB : comparer les
deux mesure surtout la dérive dans le temps, et le fichier ne contient aucun film récent.
"""
from __future__ import annotations

import shutil
import subprocess

import pandas as pd

from .config import Settings
from .normalize import clean_text, parse_tmdb_genres

DATASET = "rounakbanik/the-movies-dataset"
FILE = "movies_metadata.csv"


def download(cfg: Settings) -> bool:
    """Utilise la CLI officielle (identifiants dans ~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY)."""
    cfg.kaggle_dir.mkdir(parents=True, exist_ok=True)
    if (cfg.kaggle_dir / FILE).exists():
        print("[kaggle] déjà présent")
        return True
    if shutil.which("kaggle") is None:
        print("[kaggle] CLI absente (pip install kaggle). Alternative manuelle : télécharger "
              f"{FILE} depuis https://www.kaggle.com/datasets/{DATASET} et le placer dans {cfg.kaggle_dir}/")
        return False
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(cfg.kaggle_dir), "--unzip"],
                   check=True)
    return (cfg.kaggle_dir / FILE).exists()


def load(cfg: Settings) -> pd.DataFrame | None:
    """movies_metadata.csv nettoyé, une ligne par tmdb_id. None si le fichier est absent."""
    path = cfg.kaggle_dir / FILE
    if not path.exists():
        return None
    # dtype=str : le fichier contient des lignes décalées (id = date) qui cassent l'inférence de types.
    raw = pd.read_csv(path, dtype=str, low_memory=False)

    df = pd.DataFrame({
        "tmdb_id": pd.to_numeric(raw["id"], errors="coerce"),
        "imdb_id": raw.get("imdb_id"),
        "title": raw.get("title"),
        "overview": raw.get("overview").map(clean_text),
        "tagline": raw.get("tagline").map(clean_text),
        "genres": raw.get("genres").map(parse_tmdb_genres),
        "runtime": pd.to_numeric(raw.get("runtime"), errors="coerce"),
        "popularity": pd.to_numeric(raw.get("popularity"), errors="coerce"),
        "vote_average": pd.to_numeric(raw.get("vote_average"), errors="coerce"),
        "vote_count": pd.to_numeric(raw.get("vote_count"), errors="coerce"),
        "original_language": raw.get("original_language"),
        "release_date": raw.get("release_date"),
        "poster_path": raw.get("poster_path"),
    })
    df = df[df["tmdb_id"].notna()].copy()  # écarte les lignes malformées
    df["tmdb_id"] = df["tmdb_id"].astype("int64")
    # doublons connus : on garde la ligne la plus renseignée (plus grand nombre de votes)
    df = (df.sort_values("vote_count", ascending=False, na_position="last")
            .drop_duplicates("tmdb_id", keep="first")
            .sort_values("tmdb_id")
            .reset_index(drop=True))
    return df
