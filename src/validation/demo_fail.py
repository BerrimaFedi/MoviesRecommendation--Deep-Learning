"""Démonstration de l'« échec contrôlé » d'un test de qualité (soutenance, CI).

Copie les données dans un dossier temporaire, y injecte des défauts, puis exécute la vraie étape de
validation sur la copie : elle doit échouer avec le code 1, exactement comme dans `dvc repro`.
Les données d'origine ne sont jamais modifiées.

  python -m src.validation.demo_fail                    # sources brutes : note à 7★, doublon, film orphelin
  python -m src.validation.demo_fail --kind leakage     # tables finales : film froid recopié dans le train
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ..common.config import deep_copy, load_params
from ..common.stage import run_stage
from . import processed, raw


def corrupt_raw(params: dict, tmp: Path) -> dict:
    p = deep_copy(params)
    dst = tmp / "movielens"
    shutil.copytree(params["paths"]["raw_movielens"], dst)
    r = pd.read_csv(dst / "ratings.csv")
    r.loc[0, "rating"] = 7.0                                   # hors domaine [0,5 ; 5]
    r = pd.concat([r, r.iloc[[1]], pd.DataFrame([{"userId": 1, "movieId": 99_999_999, "rating": 4.0,
                                                  "timestamp": int(r["timestamp"].max())}])], ignore_index=True)
    r.to_csv(dst / "ratings.csv", index=False)
    p["paths"]["raw_movielens"] = str(dst)
    print("défauts injectés dans ratings.csv : note = 7.0 ; doublon (user, film) ; film absent du catalogue")
    return p


def corrupt_leakage(params: dict, tmp: Path) -> dict:
    p = deep_copy(params)
    dst = tmp / "processed"
    shutil.copytree(params["paths"]["processed"], dst)
    r = pd.read_parquet(dst / "ratings.parquet")
    cold = r.index[r["split"] == "cold_item_test"][:10]
    r.loc[cold, "split"] = "train"                             # fuite : film froid vu à l'entraînement
    r.to_parquet(dst / "ratings.parquet", index=False)
    p["paths"]["processed"] = str(dst)
    print("défaut injecté : 10 notes de films « froids » déplacées dans le train")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--kind", choices=["raw", "leakage"], default="raw")
    a = ap.parse_args()
    params = load_params(a.params)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = corrupt_raw(params, tmp) if a.kind == "raw" else corrupt_leakage(params, tmp)
        p["paths"]["reports"] = str(tmp / "reports")          # n'écrase pas les rapports réels
        stage = raw.stage if a.kind == "raw" else processed.stage
        code = run_stage(f"demo_fail_{a.kind}", stage, params=p)
    print(f"code de sortie de l'étape : {code} ({'arrêt du pipeline, comme attendu' if code else 'AUCUN échec ?!'})")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
