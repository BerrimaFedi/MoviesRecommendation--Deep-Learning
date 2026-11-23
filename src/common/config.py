"""Lecture de configs/params.yaml, surcharges en ligne de commande et réglages d'ingestion."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS = Path("configs/params.yaml")


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


def _coerce(value: str) -> Any:
    """'0.5' -> 0.5, 'true' -> True, 'null' -> None, '[1, 2]' -> [1, 2] (syntaxe YAML)."""
    return yaml.safe_load(value)


def set_path(params: dict[str, Any], dotted: str, value: Any) -> None:
    node = params
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_path(params: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = params
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lit un YAML ; la clé `extends: autre.yaml` (relative au fichier) fusionne d'abord la base."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("extends", None)
    return _deep_merge(_read_yaml(path.parent / base), data) if base else data


def load_params(path: str | Path = DEFAULT_PARAMS, overrides: list[str] | None = None) -> dict[str, Any]:
    """Charge le YAML (avec héritage `extends`) puis applique des surcharges « clé.pointée=valeur »."""
    params = _read_yaml(Path(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Surcharge invalide (attendu clé=valeur) : {item}")
        key, value = item.split("=", 1)
        set_path(params, key.strip(), _coerce(value))
    return params


def params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


class MissingCredentials(RuntimeError):
    pass


@dataclass
class Settings:
    """Réglages des étapes d'ingestion, construits depuis params.yaml (voir `from_params`)."""

    movielens_dir: Path = Path("data/raw/ml-latest-small")
    tmdb_dir: Path = Path("data/raw/tmdb")
    posters_dir: Path = Path("data/raw/posters")
    kaggle_dir: Path = Path("data/raw/kaggle")

    movielens_version: str = "ml-latest-small"
    movielens_base_url: str = "https://files.grouplens.org/datasets/movielens"

    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_image_base: str = "https://image.tmdb.org/t/p/"
    language: str = "en-US"
    poster_size: str = "w342"
    rate_per_sec: float = 20.0
    concurrency: int = 16
    max_retries: int = 6
    request_timeout_s: float = 30.0
    cache_max_age_days: int = 180
    image_min_side: int = 16

    kaggle_dataset: str = "rounakbanik/the-movies-dataset"
    kaggle_file: str = "movies_metadata.csv"

    @classmethod
    def from_params(cls, p: dict[str, Any]) -> "Settings":
        t, paths = p["tmdb"], p["paths"]
        cfg = cls(
            movielens_dir=Path(paths["raw_movielens"]), tmdb_dir=Path(paths["raw_tmdb"]),
            posters_dir=Path(paths["raw_posters"]), kaggle_dir=Path(paths["raw_kaggle"]),
            movielens_version=p["movielens"]["version"], movielens_base_url=p["movielens"]["base_url"],
            tmdb_base_url=t["base_url"], tmdb_image_base=t["image_base"], language=t["language"],
            poster_size=t["poster_size"], rate_per_sec=float(t["rate_per_sec"]),
            concurrency=int(t["concurrency"]), max_retries=int(t["max_retries"]),
            request_timeout_s=float(t["request_timeout_s"]), cache_max_age_days=int(t["cache_max_age_days"]),
            image_min_side=int(p["preprocessing"]["image"]["min_side"]),
            kaggle_dataset=p["kaggle"]["dataset"], kaggle_file=p["kaggle"]["file"],
        )
        # surcharges facultatives (serveur miroir, tests) ; par défaut les adresses officielles
        cfg.tmdb_base_url = os.environ.get("TMDB_BASE_URL", cfg.tmdb_base_url)
        cfg.tmdb_image_base = os.environ.get("TMDB_IMAGE_BASE", cfg.tmdb_image_base)
        return cfg

    @classmethod
    def for_data_dir(cls, data_dir: str | Path, movielens_version: str = "ml-latest-small",
                     **kw: Any) -> "Settings":
        raw = Path(data_dir) / "raw"
        return cls(movielens_dir=raw / movielens_version, tmdb_dir=raw / "tmdb", posters_dir=raw / "posters",
                   kaggle_dir=raw / "kaggle", movielens_version=movielens_version, **kw)

    @property
    def tmdb_cache(self) -> Path:
        return self.tmdb_dir / "movies.jsonl"

    @property
    def tmdb_config_file(self) -> Path:
        return self.tmdb_dir / "configuration.json"

    @property
    def posters_manifest(self) -> Path:
        return self.posters_dir / "manifest.jsonl"

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


def deep_copy(params: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(params)


def materialize(src: str | Path, out: str | Path) -> None:
    """Écrit la configuration complète (héritage résolu) : DVC lit un fichier sans `extends`."""
    params = load_params(src)
    header = f"# FICHIER GÉNÉRÉ depuis {Path(src).as_posix()} par `python -m src.common.config` — ne pas éditer.\n"
    Path(out).write_text(header + yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Résout l'héritage d'un fichier de paramètres")
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    materialize(a.src, a.out)
    print(f"{a.src} -> {a.out}")
