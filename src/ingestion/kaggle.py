"""Étape `ingest_kaggle` : « The Movies Dataset », source de secours hors ligne et point de comparaison.

À savoir (vérifié) : ses notes proviennent de MovieLens et ses métadonnées sont un instantané TMDB
(films sortis jusqu'en juillet 2017). Ce n'est donc PAS une source indépendante de TMDB : comparer les
deux mesure surtout la dérive dans le temps, et le fichier ne contient aucun film récent.
"""
from __future__ import annotations

import shutil
import subprocess

import pandas as pd

from ..common.config import Settings
from ..common.stage import StageContext, main_for
from ..preprocessing.normalize import clean_text, parse_tmdb_genres

UNAVAILABLE = "UNAVAILABLE.txt"  # marqueur explicite : la source facultative est absente


def download(cfg: Settings, log=print) -> bool:
    """CLI officielle (identifiants dans ~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY)."""
    cfg.kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.kaggle_dir / cfg.kaggle_file
    if target.exists():
        log("déjà présent")
        return True
    if shutil.which("kaggle") is None:
        log("CLI absente (pip install kaggle). Alternative manuelle : télécharger "
            f"{cfg.kaggle_file} depuis https://www.kaggle.com/datasets/{cfg.kaggle_dataset} "
            f"et le placer dans {cfg.kaggle_dir}/")
        return False
    subprocess.run(["kaggle", "datasets", "download", "-d", cfg.kaggle_dataset, "-p", str(cfg.kaggle_dir),
                    "--unzip"], check=True)
    return target.exists()


def load(cfg: Settings) -> pd.DataFrame | None:
    """movies_metadata.csv nettoyé, une ligne par tmdb_id. None si le fichier est absent."""
    path = cfg.kaggle_dir / cfg.kaggle_file
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


def stage(ctx: StageContext) -> None:
    cfg = Settings.from_params(ctx.params)
    cfg.kaggle_dir.mkdir(parents=True, exist_ok=True)
    marker = cfg.kaggle_dir / UNAVAILABLE
    if not ctx.params["kaggle"]["enabled"]:
        ok = False
    elif ctx.params["ingestion"]["mode"] == "synthetic":
        from .synthetic import make_kaggle
        make_kaggle(ctx.params)
        ok = True
    else:
        ok = download(cfg, log=ctx.info)
    if ok:
        marker.unlink(missing_ok=True)
        k = load(cfg)
        ctx.metric("available", True)
        ctx.metric("rows_clean", 0 if k is None else len(k))
    else:
        marker.write_text("Source Kaggle désactivée ou indisponible : secours ignoré.\n", encoding="utf-8")
        ctx.warn("source Kaggle indisponible : la collecte continue sans secours")
        ctx.metric("available", False)


if __name__ == "__main__":
    main_for("ingest_kaggle", stage, "Récupère le dataset Kaggle (facultatif)")
