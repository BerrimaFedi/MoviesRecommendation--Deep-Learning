"""Fixtures partagées : paramètres pointant vers un dossier temporaire, pipeline synthétique."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import load_params, set_path

ROOT = Path(__file__).resolve().parents[1]


def params_for(tmp: Path, base: str = "configs/params.yaml", raw_movielens: str = "movielens") -> dict:
    p = load_params(ROOT / base)
    for key, sub in {"raw_movielens": f"raw/{raw_movielens}", "raw_tmdb": "raw/tmdb", "raw_posters": "raw/posters",
                     "raw_kaggle": "raw/kaggle", "interim": "interim", "processed": "processed",
                     "features": "features", "reports": "reports", "logs": "logs"}.items():
        set_path(p, f"paths.{key}", str(tmp / sub))
    set_path(p, "paths.inbox", str(tmp / "inbox"))
    (tmp / "inbox").mkdir(parents=True, exist_ok=True)
    set_path(p, "split.mode", "recompute")
    set_path(p, "split.frozen_spec", str(tmp / "split_spec.json"))
    set_path(p, "drift.reference", str(tmp / "reports" / "reference_profile.json"))
    set_path(p, "tracking.backend", "json")
    set_path(p, "tracking.tracking_uri", str(tmp / "mlruns"))
    return p


@pytest.fixture(scope="session")
def sample_pipeline(tmp_path_factory):
    """Pipeline complet (mode synthétique) exécuté une fois pour toute la session de tests."""
    from src.pipeline import run

    tmp = tmp_path_factory.mktemp("sample")
    params = params_for(tmp, "configs/params.ci.yaml")
    assert run(params) == 0
    return params
