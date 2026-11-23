"""Tests de données : la porte de qualité laisse passer des données saines et arrête les défauts injectés."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.stage import run_stage
from src.validation import processed as vp
from src.validation import raw as vr


def _load(params):
    d = Path(params["paths"]["processed"])
    return pd.read_parquet(d / "movies.parquet"), pd.read_parquet(d / "ratings.parquet")


def _failed(suite) -> set[str]:
    return {r.name for r in suite.errors}


def test_clean_sample_passes_every_critical_check(sample_pipeline):
    m, r = _load(sample_pipeline)
    suite = vp.validate_processed(sample_pipeline, m, r)
    assert suite.errors == []
    cats = {c.category for c in suite.results}
    assert {"schema", "missing", "duplicates", "outliers", "inter_modal", "temporal", "leakage"} <= cats


def test_duplicate_ratings_are_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    r = pd.concat([r, r.head(5)], ignore_index=True)
    assert "doublons: (user_id, movie_id)" in _failed(vp.validate_processed(sample_pipeline, m, r))


def test_cold_item_leaking_into_train_is_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    cold = r.index[r["split"] == "cold_item_test"][:3]
    r.loc[cold, "split"] = "train"
    failed = _failed(vp.validate_processed(sample_pipeline, m, r))
    assert "fuite: films froids absents de train/val/test" in failed or "fuite: indicateur item_in_train exact" in failed


def test_test_rating_older_than_train_is_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    i = r.index[r["split"] == "test"][0]
    r.loc[i, "timestamp"] = r["timestamp"].min() - 1
    assert "fuite: train < val < test (chronologique)" in _failed(vp.validate_processed(sample_pipeline, m, r))


def test_inconsistent_modality_mask_is_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    i = m.index[m["has_poster"] == 1][0]
    m.loc[i, ["poster_file", "poster_sha1"]] = None  # affiche annoncée présente mais absente
    failed = _failed(vp.validate_processed(sample_pipeline, m, r))
    assert "cohérence: has_poster=1 ⇒ fichier + empreinte + dimensions" in failed


def test_missing_rate_above_threshold_is_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    m.loc[m.index[:20], "overview"] = None
    m.loc[m.index[:20], "has_overview"] = 0
    assert "manquants: overview" in _failed(vp.validate_processed(sample_pipeline, m, r))


def test_orphan_join_key_is_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    r.loc[r.index[0], "movie_id"] = 999_999
    assert "clés: ratings.movie_id ∈ movies.movie_id" in _failed(vp.validate_processed(sample_pipeline, m, r))


def test_text_encoding_defects_are_caught(sample_pipeline):
    m, r = _load(sample_pipeline)
    i = m.index[m["has_overview"] == 1]
    m.loc[i[0], "overview"] = "bad \x07 control character in a long enough synopsis"
    m.loc[i[1], "overview"] = "Café non normalisé NFC, avec assez de caractères"
    failed = _failed(vp.validate_processed(sample_pipeline, m, r))
    assert "texte: caractères de contrôle" in failed and "texte: forme Unicode NFC" in failed


def test_raw_rating_out_of_domain_stops_the_pipeline(sample_pipeline, tmp_path):
    """Démo « échec contrôlé » : une note à 7 étoiles dans la source brute -> code de sortie 1."""
    import shutil

    params = {**sample_pipeline, "paths": dict(sample_pipeline["paths"])}
    src = Path(sample_pipeline["paths"]["raw_movielens"])
    bad = tmp_path / "movielens_corrupt"
    shutil.copytree(src, bad)
    ratings = pd.read_csv(bad / "ratings.csv")
    ratings.loc[0, "rating"] = 7.0
    ratings.to_csv(bad / "ratings.csv", index=False)
    params["paths"]["raw_movielens"] = str(bad)
    params["paths"]["reports"] = str(tmp_path / "reports")
    assert "movielens.ratings.rating ≤ 5.0" in _failed(vr.validate_raw(params))
    assert run_stage("validate_raw", vr.stage, params=params) == 1


@pytest.mark.parametrize("stage_mod", [vr, vp])
def test_validation_stages_write_their_report(sample_pipeline, stage_mod):
    name = "raw" if stage_mod is vr else "processed"
    assert (Path(sample_pipeline["paths"]["reports"]) / "validation" / f"{name}.json").exists()
