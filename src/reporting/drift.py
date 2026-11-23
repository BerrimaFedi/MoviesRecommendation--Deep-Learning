"""Profil des données et détection de dérive entre deux versions (bonus).

Le profil (reports/profile.json, suivi comme métrique DVC et versionné par Git) contient des
histogrammes à bornes FIXES par variable : deux profils sont donc toujours comparables, et l'indice de
stabilité de population (PSI) se calcule sans relire les données de la version précédente.

    PSI = Σ_b (p_b − q_b) · ln(p_b / q_b)     (q : référence, p : version courante)
    < 0,10 stable · 0,10–0,25 dérive modérée · > 0,25 dérive forte

Utilisation :
  python -m src.reporting.drift --ref reports/reference_profile.json      # contre la référence figée
  python -m src.reporting.drift --ref-rev task3                           # contre une étiquette Git
  python -m src.reporting.drift --promote                                 # le profil courant devient la référence
`dvc metrics diff` donne en complément l'écart des statistiques scalaires entre deux commits.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import load_params
from ..common.io import read_json, write_json

EPS = 1e-4
NUMERIC_BINS: dict[str, list[float]] = {
    "release_year": list(range(1900, 2036, 5)),
    "runtime_min": [0, 60, 75, 85, 90, 95, 100, 105, 110, 120, 130, 150, 180, 240, 10_000],
    "overview_n_words": [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 10_000],
    "log_popularity": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 100],
    "poster_aspect": [0, 1.3, 1.45, 1.49, 1.51, 1.55, 1.7, 10],
}
RATING_BINS = [0.25 + 0.5 * i for i in range(11)]
COUNT_BINS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 10**9]


def _hist(values: pd.Series, edges: list[float]) -> list[float]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), bins=edges)
    total = counts.sum()
    return (counts / total).round(6).tolist() if total else [0.0] * (len(edges) - 1)


def _dist(values: pd.Series) -> dict[str, float]:
    return values.value_counts(normalize=True).round(6).sort_index().to_dict()


def build_profile(movies: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, Any]:
    m = movies.assign(poster_aspect=movies["poster_h"] / movies["poster_w"])
    per_item = ratings.groupby("movie_id").size()
    per_user = ratings.groupby("user_id").size()
    return {
        "volumes": {"movies": len(m), "ratings": len(ratings), "users": int(ratings["user_id"].nunique()),
                    "density": round(len(ratings) / max(1, ratings["user_id"].nunique() * ratings["movie_id"].nunique()), 6)},
        "coverage": {c: round(float(m[c].mean()), 6) for c in
                     ("has_overview", "has_poster", "has_meta", "all_modalities", "runtime_missing", "year_missing")},
        "histograms": {
            **{c: {"edges": e, "p": _hist(m[c], e)} for c, e in NUMERIC_BINS.items()},
            "rating": {"edges": RATING_BINS, "p": _hist(ratings["rating"], RATING_BINS)},
            "ratings_per_item": {"edges": COUNT_BINS, "p": _hist(per_item, COUNT_BINS)},
            "ratings_per_user": {"edges": COUNT_BINS, "p": _hist(per_user, COUNT_BINS)},
        },
        "categorical": {
            "lang_cat": _dist(m["lang_cat"]),
            "split": _dist(ratings["split"]),
            "overview_source": _dist(m["overview_source"].fillna("none")),
            "modality_mask": _dist(m["modality_mask"]),
            "genres": {c[2:]: round(float(m[c].mean()), 6) for c in m.columns if c.startswith("g_")},
        },
        "sources": {
            "movies": m["source"].value_counts().to_dict() if "source" in m else {"movielens": len(m)},
            "ratings": ratings["source"].value_counts().to_dict() if "source" in ratings else {"movielens": len(ratings)},
            "ratings_by_split": ({s: g["split"].value_counts().to_dict() for s, g in ratings.groupby("source")}
                                 if "source" in ratings else {}),
        },
        "summary": {
            "rating_mean": round(float(ratings["rating"].mean()), 4),
            "overview_words_median": float(m.loc[m["has_overview"] == 1, "overview_n_words"].median()),
            "ratings_per_item_median": float(per_item.median()),
            "items_with_at_most_5_ratings": round(float((per_item <= 5).mean()), 4),
            "genre_mean_jaccard": round(float(m["genre_agreement"].mean()), 4),
        },
    }


def psi(p: list[float], q: list[float]) -> float:
    a, b = np.asarray(p, float) + EPS, np.asarray(q, float) + EPS
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((a - b) * np.log(a / b)))


def compare(ref: dict[str, Any], cur: dict[str, Any], dp: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def level(x: float) -> str:
        return "alert" if x > dp["psi_alert"] else "warn" if x > dp["psi_warn"] else "ok"

    for name, h in cur["histograms"].items():
        r = ref.get("histograms", {}).get(name)
        if not r or r["edges"] != h["edges"]:
            continue
        v = psi(h["p"], r["p"])
        rows.append({"variable": name, "kind": "psi", "value": round(v, 4), "level": level(v)})
    for name, dist in cur["categorical"].items():
        r = ref.get("categorical", {}).get(name)
        if r is None or name == "genres":
            continue
        keys = sorted(set(dist) | set(r))
        v = psi([dist.get(k, 0) for k in keys], [r.get(k, 0) for k in keys])
        rows.append({"variable": name, "kind": "psi", "value": round(v, 4), "level": level(v)})
    for name, val in cur["coverage"].items():
        delta = val - ref.get("coverage", {}).get(name, val)
        rows.append({"variable": name, "kind": "coverage_delta", "value": round(delta, 4),
                     "level": "alert" if abs(delta) > dp["coverage_alert"] else "ok"})
    vol = {k: {"ref": ref.get("volumes", {}).get(k), "cur": v} for k, v in cur["volumes"].items()}
    worst = "alert" if any(r["level"] == "alert" for r in rows) else (
        "warn" if any(r["level"] == "warn" for r in rows) else "ok")
    return {"status": worst, "volumes": vol, "rows": rows}


def profile_at_rev(rev: str, path: str) -> dict[str, Any]:
    out = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dérive entre deux profils de données")
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--ref", help="profil de référence (fichier)")
    ap.add_argument("--ref-rev", help="révision Git (commit, étiquette) dont on lit reports/profile.json")
    ap.add_argument("--cur", help="profil courant (défaut : <reports>/profile.json)")
    ap.add_argument("--promote", action="store_true", help="copie le profil courant vers drift.reference")
    a = ap.parse_args()
    p = load_params(a.params)
    reports = Path(p["paths"]["reports"])
    cur_path = Path(a.cur) if a.cur else reports / "profile.json"
    if a.promote:
        shutil.copyfile(cur_path, p["drift"]["reference"])
        print(f"{cur_path} -> {p['drift']['reference']} (nouvelle référence)")
        return
    ref = profile_at_rev(a.ref_rev, (reports / "profile.json").as_posix()) if a.ref_rev else read_json(
        Path(a.ref or p["drift"]["reference"]))
    res = compare(ref, read_json(cur_path), p["drift"])
    write_json(reports / "drift_adhoc.json", res)
    for r in res["rows"]:
        if r["level"] != "ok":
            print(f"  {r['level']:5s} {r['variable']:20s} {r['kind']}={r['value']}")
    print(f"dérive : {res['status']} ({len(res['rows'])} indicateurs) -> {reports / 'drift_adhoc.json'}")


if __name__ == "__main__":
    main()
