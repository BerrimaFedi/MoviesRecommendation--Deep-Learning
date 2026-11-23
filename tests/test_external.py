"""Données apportées par un tiers (inbox) et découpage figé : intégration, contrôles, stabilité de l'évaluation."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import set_path
from src.common.stage import run_stage
from src.ingestion import external
from src.pipeline import run
from src.splitting import split
from src.splitting.freeze import compute_spec
from tests.conftest import params_for


@pytest.fixture(scope="module")
def frozen_run(tmp_path_factory):
    """Échantillon complet, gel du découpage, puis dépôt d'un jeu externe valide et ré-exécution."""
    tmp = tmp_path_factory.mktemp("ext")
    p = params_for(tmp, "configs/params.ci.yaml")
    assert run(p) == 0
    before = pd.read_parquet(Path(p["paths"]["processed"]) / "ratings.parquet")
    interim = Path(p["paths"]["interim"])
    spec = compute_spec(p, pd.read_parquet(interim / "ratings.parquet"), pd.read_parquet(interim / "movies_aligned.parquet"))
    from src.common.io import write_json

    write_json(Path(p["split"]["frozen_spec"]), spec)
    set_path(p, "split.mode", "frozen")
    assert run(p, start="split") == 0
    frozen_base = pd.read_parquet(Path(p["paths"]["processed"]) / "ratings.parquet")
    external.write_example(Path(p["paths"]["inbox"]))
    assert run(p, start="ingest_external") == 0
    return {"params": p, "before": before, "frozen_base": frozen_base,
            "after": pd.read_parquet(Path(p["paths"]["processed"]) / "ratings.parquet"),
            "movies": pd.read_parquet(Path(p["paths"]["processed"]) / "movies.parquet")}


def test_frozen_spec_reproduces_recompute(frozen_run):
    a, b = frozen_run["before"], frozen_run["frozen_base"]
    assert a["split"].tolist() == b["split"].tolist()


def test_external_ratings_go_to_new_segment_and_base_is_untouched(frozen_run):
    after = frozen_run["after"]
    ext = after[after["source"] == "external"]
    assert len(ext) == 5 and set(ext["split"]) == {"new"}
    base = after[after["source"] == "movielens"].reset_index(drop=True)
    assert base["split"].tolist() == frozen_run["frozen_base"]["split"].tolist()


def test_external_movies_are_aligned_with_all_modalities(frozen_run):
    m = frozen_run["movies"].set_index("movie_id")
    assert m.loc[900001, "source"] == "external" and m.loc[900001, "modality_mask"] == "TIM"
    assert list(m.loc[900001, "genres_unified"]) == ["Adventure", "Science Fiction"]
    assert m.loc[900002, "original_language"] == "fr" and m.loc[900002, "g_romance"] == 1


def test_external_movies_get_features(frozen_run):
    p = frozen_run["params"]
    z = np.load(Path(p["paths"]["features"]) / "items.npz")
    rows = np.isin(z["movie_id"], [900001, 900002])
    assert rows.sum() == 2 and (z["mask"][rows] == 1).all()
    assert np.all(np.linalg.norm(z["image"][rows], axis=1) > 0)


def test_dashboard_shows_external_source(frozen_run):
    html = (Path(frozen_run["params"]["paths"]["reports"]) / "quality_report.html").read_text(encoding="utf-8")
    assert "Sources des données" in html and "external" in html


def test_defective_inbox_stops_the_pipeline(frozen_run, tmp_path):
    p = copy.deepcopy(frozen_run["params"])
    set_path(p, "paths.inbox", str(tmp_path / "bad_inbox"))
    set_path(p, "paths.interim", str(tmp_path / "interim"))
    set_path(p, "paths.reports", str(tmp_path / "reports"))
    external.write_example(tmp_path / "bad_inbox", bad=True)
    _, _, suite, _ = external.build_external(p)
    failed = {r.name for r in suite.errors}
    assert {"movies.csv: movie_id absent de MovieLens (pas de collision)",
            "posters/: affiches déclarées présentes et décodables",
            "ratings.csv: movie_id connu (MovieLens ou movies.csv)"} <= failed
    assert any(n.startswith("ratings.csv: note ∈") for n in failed)
    assert run_stage("ingest_external", external.stage, params=p) == 1


def test_recompute_mode_integrates_external_ratings(sample_pipeline, tmp_path):
    p = params_for(tmp_path, "configs/params.ci.yaml")
    assert run(p, stop="ingest_posters") == 0
    external.write_example(Path(p["paths"]["inbox"]))
    assert run(p, start="ingest_external", stop="split") == 0
    r = pd.read_parquet(Path(p["paths"]["processed"]) / "ratings.parquet")
    assert "new" not in set(r["split"])
    assert set(r.loc[r["source"] == "external", "split"]) <= {"test", "cold_item_test"}  # notes 2025 = plus récentes


def test_empty_inbox_changes_nothing(sample_pipeline):
    r = pd.read_parquet(Path(sample_pipeline["paths"]["processed"]) / "ratings.parquet")
    assert set(r["source"]) == {"movielens"}


def test_genre_and_timestamp_parsing():
    unknown: dict[str, int] = {}
    assert external.parse_genres("Sci-Fi|Children|Science Fiction|Telenovela|IMAX", unknown) == [
        "Family", "Science Fiction"]
    assert unknown == {"Telenovela": 1}
    ts = external.parse_timestamp(pd.Series(["964982703", "2025-06-02T10:00:00Z", "pas une date"]))
    assert ts.iloc[0] == 964982703 and ts.iloc[1] == 1748858400 and pd.isna(ts.iloc[2])


def test_spec_parameter_mismatch_is_refused(tmp_path):
    p = params_for(tmp_path)
    from src.common.io import write_json

    write_json(tmp_path / "split_spec.json", {"seed": 1, "train_frac": 0.8, "val_frac": 0.1, "cold_item_frac": 0.05,
                                              "cold_item_min_ratings": 5})
    with pytest.raises(ValueError, match="seed"):
        split.load_spec(p)
