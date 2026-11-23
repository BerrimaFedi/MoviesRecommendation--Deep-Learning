"""Exécute toutes les étapes dans l'ordre du DAG, sans DVC (CI rapide, échantillon, débogage).

La commande de référence reste `dvc repro` (cache, ré-exécution sélective). Ce lanceur sert à :
  python -m src.pipeline --params configs/params.ci.yaml            # pipeline complet sur l'échantillon
  python -m src.pipeline --from preprocess_text --to validate_processed
Il s'arrête à la première étape en échec (même règle que DVC).
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .common.config import load_dotenv, load_params
from .common.stage import run_stage

# (nom de l'étape DVC, module) — même ordre topologique que dvc.yaml
STAGES: list[tuple[str, str]] = [
    ("ingest_movielens", "src.ingestion.movielens"),
    ("ingest_kaggle", "src.ingestion.kaggle"),
    ("ingest_tmdb", "src.ingestion.tmdb"),
    ("ingest_posters", "src.ingestion.posters"),
    ("ingest_external", "src.ingestion.external"),
    ("validate_raw", "src.validation.raw"),
    ("preprocess_catalog", "src.preprocessing.catalog"),
    ("preprocess_text", "src.preprocessing.text"),
    ("preprocess_images", "src.preprocessing.image"),
    ("preprocess_meta", "src.preprocessing.tabular"),
    ("preprocess_ratings", "src.preprocessing.ratings"),
    ("align", "src.alignment.align"),
    ("split", "src.splitting.split"),
    ("validate_processed", "src.validation.processed"),
    ("extract_features", "src.features.extract"),
    ("report", "src.reporting.report"),
]


def run(params: dict, start: str | None = None, stop: str | None = None) -> int:
    names = [n for n, _ in STAGES]
    i0 = names.index(start) if start else 0
    i1 = names.index(stop) if stop else len(names) - 1
    for name, module in STAGES[i0:i1 + 1]:
        fn = importlib.import_module(module).stage
        code = run_stage(name, fn, params=params)
        if code != 0:
            print(f"[pipeline] arrêt : l'étape {name} a échoué (code {code})", file=sys.stderr)
            return code
    print("[pipeline] terminé sans erreur")
    return 0


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--from", dest="start", choices=[n for n, _ in STAGES])
    ap.add_argument("--to", dest="stop", choices=[n for n, _ in STAGES])
    a = ap.parse_args()
    raise SystemExit(run(load_params(a.params, a.overrides), a.start, a.stop))


if __name__ == "__main__":
    main()
