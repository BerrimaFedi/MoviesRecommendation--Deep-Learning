"""Tests DataOps : configuration, reproductibilité, garde-fou synthétique, dérive, traçabilité."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.common.config import load_params, materialize
from src.common.io import read_json
from src.common.stage import run_stage
from src.ingestion.synthetic import RefuseOverwrite, make_movielens
from src.reporting.drift import build_profile, compare
from src.splitting import split
from tests.conftest import ROOT, params_for


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overrides_and_inheritance():
    p = load_params(ROOT / "configs/params.ci.yaml", ["split.train_frac=0.7", "tmdb.limit=null"])
    assert p["ingestion"]["mode"] == "synthetic"           # surcharge du fichier CI
    assert p["tmdb"]["language"] == "en-US"                 # hérité de params.yaml
    assert p["split"]["train_frac"] == 0.7 and p["tmdb"]["limit"] is None


def test_materialized_ci_params_are_complete(tmp_path):
    out = tmp_path / "params.yaml"
    materialize(ROOT / "configs/params.ci.yaml", out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "extends" not in data and data["paths"]["raw_movielens"].startswith("data_sample")
    assert set(data) == set(load_params(ROOT / "configs/params.yaml"))


def test_no_hardcoded_data_paths_in_sources():
    """Les chemins de données ne viennent que de la configuration."""
    offenders = [f for f in (ROOT / "src").rglob("*.py")
                 if any(s in f.read_text(encoding="utf-8") for s in ('"data/raw', "'data/raw", '"data/processed'))
                 and f.name != "config.py"]
    assert offenders == []


def test_split_is_byte_for_byte_reproducible(sample_pipeline, tmp_path):
    params = {**sample_pipeline, "paths": {**sample_pipeline["paths"], "processed": str(tmp_path / "p2"),
                                           "reports": str(tmp_path / "r2")}}
    assert run_stage("split", split.stage, params=params) == 0
    a, b = Path(sample_pipeline["paths"]["processed"]), tmp_path / "p2"
    for f in ("ratings.parquet", "movies.parquet", "manifest.json"):
        assert _sha(a / f) == _sha(b / f), f


def test_seed_changes_the_cold_items(sample_pipeline):
    m = pd.read_parquet(Path(sample_pipeline["paths"]["interim"]) / "movies_aligned.parquet")
    r = pd.read_parquet(Path(sample_pipeline["paths"]["interim"]) / "ratings.parquet")
    other = {**sample_pipeline, "project": {"seed": 7}}
    c1 = set(split.split_ratings(sample_pipeline, r, m)[0].query("split == 'cold_item_test'")["movie_id"])
    c2 = set(split.split_ratings(other, r, m)[0].query("split == 'cold_item_test'")["movie_id"])
    assert c1 != c2


def test_synthetic_generator_refuses_to_overwrite_real_data(tmp_path):
    p = params_for(tmp_path)
    d = Path(p["paths"]["raw_movielens"])
    d.mkdir(parents=True)
    (d / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,1,4.0,964982703\n", encoding="utf-8")
    with pytest.raises(RefuseOverwrite):
        make_movielens(p)


def test_every_stage_records_observability_metrics(sample_pipeline):
    from src.pipeline import STAGES

    stages = Path(sample_pipeline["paths"]["reports"]) / "stages"
    for name, _ in STAGES:
        rec = read_json(stages / f"{name}.json")
        assert rec["status"] == "ok" and rec["duration_s"] >= 0 and rec["params_hash"]
    assert read_json(stages / "preprocess_ratings.json")["metrics"]["ratings_kept"] > 0
    log = Path(sample_pipeline["paths"]["logs"]) / "pipeline.jsonl"
    first = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert {"ts", "level", "stage", "msg"} <= set(first)


def test_drift_is_zero_on_itself_and_detected_on_shift(sample_pipeline):
    d = Path(sample_pipeline["paths"]["processed"])
    m, r = pd.read_parquet(d / "movies.parquet"), pd.read_parquet(d / "ratings.parquet")
    ref = build_profile(m, r)
    assert compare(ref, ref, sample_pipeline["drift"])["status"] == "ok"
    shifted = m.assign(runtime_min=m["runtime_min"] * 2.5, release_year=m["release_year"] - 40)
    res = compare(ref, build_profile(shifted, r), sample_pipeline["drift"])
    alerts = {x["variable"] for x in res["rows"] if x["level"] == "alert"}
    assert res["status"] == "alert" and {"runtime_min", "release_year"} <= alerts


def test_html_report_and_data_card(sample_pipeline):
    rep = Path(sample_pipeline["paths"]["reports"])
    html = (rep / "quality_report.html").read_text(encoding="utf-8")
    assert "Tableau de bord du pipeline" in html and "validate_processed" in html
    card = (rep / "data_card.md").read_text(encoding="utf-8")
    assert "Biais et limites" in card and "Sources, licences" in card
