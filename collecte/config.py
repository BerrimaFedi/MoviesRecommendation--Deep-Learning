"""Configuration centrale du pipeline de collecte (Projet 10)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Charge un fichier .env minimal (KEY=VALUE) sans écraser l'environnement existant."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class MissingCredentials(RuntimeError):
    pass


@dataclass
class Settings:
    # --- Emplacements -------------------------------------------------------
    data_dir: Path = Path("data")

    # --- MovieLens ----------------------------------------------------------
    # ml-latest-small : développement. ml-25m / ml-32m : résultats finaux (versions stables).
    movielens_version: str = "ml-latest-small"
    movielens_base_url: str = "https://files.grouplens.org/datasets/movielens"

    # --- TMDB ---------------------------------------------------------------
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_image_base: str = "https://image.tmdb.org/t/p/"  # écrasé par /configuration si joignable
    language: str = "en-US"  # une seule langue de synopsis pour toute la base (cohérence)
    poster_size: str = "w342"  # 342 px de large, ratio 2:3 ; ample pour un CNN en 224 px

    # Plafond souple annoncé par TMDB : ~40 req/s. On reste nettement en dessous.
    rate_per_sec: float = 20.0
    concurrency: int = 16
    max_retries: int = 6
    request_timeout_s: float = 30.0
    # Les conditions d'API de TMDB limitent la durée de cache des données (6 mois).
    cache_max_age_days: int = 180

    # --- Construction du jeu final -----------------------------------------
    top_languages: int = 15
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1  # le reste (0.1) = test
    cold_item_frac: float = 0.05
    cold_item_min_ratings: int = 5

    # --- Chemins dérivés ----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def movielens_dir(self) -> Path:
        return self.raw_dir / self.movielens_version

    @property
    def tmdb_dir(self) -> Path:
        return self.raw_dir / "tmdb"

    @property
    def tmdb_cache(self) -> Path:
        return self.tmdb_dir / "movies.jsonl"

    @property
    def tmdb_config_file(self) -> Path:
        return self.tmdb_dir / "configuration.json"

    @property
    def posters_dir(self) -> Path:
        return self.raw_dir / "posters"

    @property
    def posters_manifest(self) -> Path:
        return self.posters_dir / "manifest.jsonl"

    @property
    def kaggle_dir(self) -> Path:
        return self.raw_dir / "kaggle"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.tmdb_dir, self.posters_dir, self.kaggle_dir,
                  self.processed_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- Identifiants TMDB (jamais dans le dépôt : variables d'environnement) ---
    def tmdb_auth(self) -> tuple[dict[str, str], dict[str, str]]:
        """Retourne (headers, params). Clé API v3 de préférence, sinon jeton v4."""
        key = os.environ.get("TMDB_API_KEY", "").strip()
        token = os.environ.get("TMDB_READ_TOKEN", "").strip()
        if key:
            return {"accept": "application/json"}, {"api_key": key}
        if token:
            return {"Authorization": f"Bearer {token}", "accept": "application/json"}, {}
        raise MissingCredentials(
            "Aucun identifiant TMDB. Définissez TMDB_READ_TOKEN (jeton v4) ou TMDB_API_KEY (clé v3) "
            "dans l'environnement ou dans un fichier .env (voir .env.example)."
        )
