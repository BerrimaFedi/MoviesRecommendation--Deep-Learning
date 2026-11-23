"""Étape `validate_processed` : porte de qualité sur les tables prêtes pour le modèle.

Familles de contrôles (catégorie dans le rapport) :
  schema       colonnes, types, domaines de movies.parquet / ratings.parquet
  modality     texte (UTF-8, NFC, caractères de contrôle, longueur), image (largeur w342, ratio, fichier),
               tabulaire (format des codes langue, multi-hot cohérent avec la liste de genres)
  missing      taux de valeurs manquantes par colonne ≤ seuil
  duplicates   movie_id, (user_id, movie_id), affiches partagées
  outliers     durée, année, popularité hors bornes
  inter_modal  masques de présence cohérents avec le contenu, clés de jointure valides, couverture
  temporal     sortie du film ≤ première note, fraîcheur de l'instantané TMDB
  leakage      films froids absents de train/val/test, ordre chronologique, aucune paire partagée
Sortie : reports/validation/processed.json ; code 1 si un contrôle critique échoue.
"""
from __future__ import annotations

import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..common.stage import StageContext, main_for
from ..preprocessing.normalize import CANON_GENRES, slug
from ..preprocessing.text import _CONTROL
from .checks import Suite
from .schemas import check_schema, movies_spec, ratings_spec


def _modality_checks(suite: Suite, m: pd.DataFrame, p: dict[str, Any]) -> None:
    v = p["validation"]["processed"]
    tp = p["preprocessing"]["text"]
    # --- texte : encodage et forme ------------------------------------------------------------------
    txt = m["overview"].dropna()

    def utf8_ok(s: str) -> bool:
        try:
            s.encode("utf-8", errors="strict")
            return True
        except UnicodeEncodeError:  # demi-paires de substitution isolées
            return False

    suite.rate_at_most("texte: encodable en UTF-8 strict", ~txt.map(utf8_ok), max_rate=0.0, category="schema")
    suite.rate_at_most(f"texte: forme Unicode {tp['unicode_form']}",
                       txt.map(lambda s: not unicodedata.is_normalized(tp["unicode_form"], s)), max_rate=0.0,
                       category="schema")
    suite.rate_at_most("texte: caractères de contrôle", txt.map(lambda s: bool(_CONTROL.search(s))),
                       max_rate=v["text_max_control_char_rate"], category="schema")
    suite.rate_at_most(f"texte: longueur ≤ {tp['max_len_chars']} caractères", txt.str.len() > tp["max_len_chars"],
                       max_rate=0.0, category="schema")
    # --- image : dimensions attendues --------------------------------------------------------------
    img = m[m["has_poster"] == 1]
    suite.rate_at_most(f"image: largeur = {v['poster_expected_width']} px (taille TMDB demandée)",
                       img["poster_w"] != v["poster_expected_width"], max_rate=0.01, category="schema",
                       severity="warn", detail="les affiches plus petites seront redimensionnées à l'extraction")
    lo, hi = v["poster_aspect_range"]
    aspect = img["poster_h"] / img["poster_w"]
    suite.rate_at_most(f"image: ratio hauteur/largeur ∈ [{lo}, {hi}]", ~aspect.between(lo, hi), max_rate=0.005,
                       category="outliers")
    # --- tabulaire ------------------------------------------------------------------------------------
    suite.rate_at_most("tabulaire: code langue ISO 639 (2-3 lettres minuscules)",
                       ~m["original_language"].str.fullmatch(r"[a-z]{2,3}"), max_rate=0.0, category="schema")
    multi_hot = np.stack([m[f"g_{slug(g)}"].to_numpy() for g in CANON_GENRES], axis=1)
    from_list = np.array([[int(g in set(L)) for g in CANON_GENRES] for L in m["genres_unified"]])
    suite.rate_at_most("tabulaire: colonnes g_* = genres_unified", (multi_hot != from_list).any(axis=1).sum(),
                       len(m), max_rate=0.0, category="schema")


def _quality_checks(suite: Suite, m: pd.DataFrame, r: pd.DataFrame, p: dict[str, Any]) -> None:
    v = p["validation"]["processed"]
    for col, thr in v["max_missing"].items():
        suite.rate_at_most(f"manquants: {col}", m[col].isna(), max_rate=thr, category="missing")
    suite.rate_at_most("doublons: movie_id", m["movie_id"].duplicated(), max_rate=v["max_duplicate_rate"],
                       category="duplicates")
    suite.rate_at_most("doublons: (user_id, movie_id)", r.duplicated(["user_id", "movie_id"]),
                       max_rate=v["max_duplicate_rate"], category="duplicates")
    shared = m.dropna(subset=["poster_sha1"]).groupby("poster_sha1")["movie_id"].nunique()
    suite.check("doublons: affiche identique partagée par plusieurs films", int((shared > 1).sum()) <= 10,
                category="duplicates", severity="warn", observed=int((shared > 1).sum()), threshold=10)
    dup_title = m.duplicated(["title", "release_year"], keep=False) & m["title"].notna()
    suite.check("doublons: même (titre, année) pour plusieurs movie_id", int(dup_title.sum()) <= 20,
                category="duplicates", severity="warn", observed=int(dup_title.sum()), threshold=20)

    year_max = datetime.now(UTC).year + v["year_max_offset"]
    mp = p["preprocessing"]["meta"]
    suite.rate_at_most(f"aberrants: année ∉ [{v['year_min']}, {year_max}]",
                       ~m["release_year"].dropna().astype(int).between(v["year_min"], year_max),
                       total=len(m), max_rate=v["max_outlier_rate"], category="outliers")
    suite.rate_at_most(f"aberrants: durée > {mp['runtime_max']} min", m["runtime_min"] > mp["runtime_max"],
                       max_rate=v["max_outlier_rate"], category="outliers")
    pop = m["log_popularity"].dropna()
    z = (pop - pop.mean()) / (pop.std() or 1.0)
    suite.rate_at_most("aberrants: log(popularité) à |z| > 6", z.abs() > 6, total=len(m),
                       max_rate=v["max_outlier_rate"], category="outliers", severity="warn")


def _inter_modal_checks(suite: Suite, m: pd.DataFrame, r: pd.DataFrame, p: dict[str, Any]) -> None:
    v = p["validation"]["processed"]
    tp = p["preprocessing"]["text"]
    s = m["has_overview"] + m["has_poster"] + m["has_meta"]
    suite.rate_at_most("cohérence: n_modalities = somme des masques", s != m["n_modalities"], max_rate=0.0,
                       category="inter_modal")
    mask = [("T" if a else "-") + ("I" if b else "-") + ("M" if c else "-")
            for a, b, c in zip(m["has_overview"], m["has_poster"], m["has_meta"])]
    suite.rate_at_most("cohérence: modality_mask = masques", pd.Series(mask, index=m.index) != m["modality_mask"],
                       max_rate=0.0, category="inter_modal")
    suite.rate_at_most("cohérence: has_overview=1 ⇒ synopsis ≥ longueur minimale",
                       (m["has_overview"] == 1) & (m["overview"].fillna("").str.len() < tp["min_len_chars"]),
                       max_rate=0.0, category="inter_modal")
    suite.rate_at_most("cohérence: has_poster=1 ⇒ fichier + empreinte + dimensions",
                       (m["has_poster"] == 1) & (m["poster_file"].isna() | m["poster_sha1"].isna() | m["poster_w"].isna()),
                       max_rate=0.0, category="inter_modal")
    suite.rate_at_most("cohérence: has_poster=0 ⇒ poster_file absent (absence explicite)",
                       (m["has_poster"] == 0) & m["poster_file"].notna(), max_rate=0.0, category="inter_modal")
    suite.rate_at_most("cohérence: sans tmdb_id ⇒ in_tmdb=0", m["tmdb_id"].isna() & (m["in_tmdb"] == 1),
                       max_rate=0.0, category="inter_modal")
    suite.rate_at_most("clés: ratings.movie_id ∈ movies.movie_id", ~r["movie_id"].isin(m["movie_id"]),
                       max_rate=0.0, category="inter_modal")
    suite.rate_at_most("clés: tmdb_id partagé par plusieurs movie_id", m["tmdb_id"].dropna().duplicated(),
                       total=len(m), max_rate=0.001, category="inter_modal", severity="warn")
    rate = float(m["all_modalities"].mean()) if len(m) else 0.0
    suite.check("couverture: films avec les trois modalités", rate >= v["min_all_modalities_rate"],
                category="inter_modal", observed=round(rate, 4), threshold=v["min_all_modalities_rate"])
    rated = m[m["movie_id"].isin(r["movie_id"])]
    suite.rate_at_most("couverture: film noté sans aucun contenu (ni texte ni image)",
                       (rated["has_overview"] == 0) & (rated["has_poster"] == 0), total=len(rated), max_rate=0.01,
                       category="inter_modal", severity="warn")
    ga = m["genre_agreement"].dropna()
    suite.check("cohérence: accord des genres MovieLens / TMDB (Jaccard moyen)",
                (ga.mean() if len(ga) else 1.0) >= v["min_genre_agreement"], category="inter_modal", severity="warn",
                observed=round(float(ga.mean()), 4) if len(ga) else None, threshold=v["min_genre_agreement"])


def _temporal_checks(suite: Suite, m: pd.DataFrame, r: pd.DataFrame, p: dict[str, Any]) -> None:
    v = p["validation"]["processed"]
    first = pd.to_datetime(r.groupby("movie_id")["timestamp"].min(), unit="s").dt.year
    rel = m.set_index("movie_id")["release_year"].reindex(first.index)
    early = (first < rel.astype("float")).fillna(False)
    suite.rate_at_most("synchronisation: année de sortie ≤ année de la première note", early, total=len(first),
                       max_rate=v["max_release_after_first_rating_rate"], category="temporal",
                       detail="une note antérieure à la sortie signale un mauvais appariement movie_id ↔ tmdb_id")
    fetched = pd.to_datetime(m["tmdb_fetched_at"], errors="coerce", utc=True).dropna()
    if len(fetched) and p["ingestion"]["mode"] != "synthetic":
        limit = datetime.now(UTC) - timedelta(days=p["tmdb"]["cache_max_age_days"])
        suite.rate_at_most("fraîcheur: instantané TMDB dans la durée de cache autorisée", fetched < limit,
                           max_rate=0.0, category="temporal", severity="warn",
                           detail="relancer ingest_tmdb : les conditions TMDB limitent le cache à 6 mois")


def _leakage_checks(suite: Suite, m: pd.DataFrame, r: pd.DataFrame, p: dict[str, Any]) -> None:
    sp = p["split"]
    cold = set(r.loc[r["split"] == "cold_item_test", "movie_id"])
    warm = r[r["split"].isin(["train", "val", "test"])]   # le segment `new` n'est ni appris ni évalué
    suite.check("fuite: films froids absents de train/val/test", not (set(warm["movie_id"]) & cold),
                category="leakage", observed=len(set(warm["movie_id"]) & cold), threshold=0)
    suite.rate_at_most("fuite: item_in_train=False pour les films froids",
                       r.loc[r["split"] == "cold_item_test", "item_in_train"], max_rate=0.0, category="leakage")
    ts = {s: r.loc[r["split"] == s, "timestamp"] for s in ("train", "val", "test")}
    if all(len(x) for x in ts.values()):
        order_ok = ts["train"].max() <= ts["val"].min() and ts["val"].max() <= ts["test"].min()
        suite.check("fuite: train < val < test (chronologique)", order_ok, category="leakage",
                    observed={k: [int(x.min()), int(x.max())] for k, x in ts.items()}, threshold="max(a) ≤ min(b)")
    pairs = r.groupby(["user_id", "movie_id"])["split"].nunique()
    suite.rate_at_most("fuite: paire (utilisateur, film) dans plusieurs ensembles", pairs > 1, max_rate=0.0,
                       category="leakage")
    train_ids = set(r.loc[r["split"] == "train", "movie_id"])
    recomputed = r["movie_id"].isin(train_ids)
    suite.rate_at_most("fuite: indicateur item_in_train exact", recomputed != r["item_in_train"], max_rate=0.0,
                       category="leakage")
    sha = m.set_index("movie_id")["poster_sha1"]
    cold_sha = set(sha.reindex(list(cold)).dropna())
    train_sha = set(sha.reindex(list(train_ids)).dropna())
    suite.check("fuite: affiche d'un film froid identique à celle d'un film d'entraînement",
                len(cold_sha & train_sha) == 0, category="leakage", severity="warn",
                observed=len(cold_sha & train_sha), threshold=0)
    n_warm = len(warm)
    if n_warm:
        share = (warm["split"] == "train").mean()
        suite.check("découpage: part de train conforme à la configuration", abs(share - sp["train_frac"]) <= 0.02,
                    category="leakage", severity="warn", observed=round(float(share), 4), threshold=sp["train_frac"])
    if sp["mode"] == "frozen":
        from ..splitting.split import assign, load_spec

        spec = load_spec(p)
        expected = assign(r.drop(columns=["split", "user_in_train", "item_in_train"]), spec)[0]
        key = ["timestamp", "user_id", "movie_id"]
        same = (expected.sort_values(key, kind="stable")["split"].to_numpy()
                == r.sort_values(key, kind="stable")["split"].to_numpy())
        suite.rate_at_most("découpage figé: affectation conforme à configs/split_spec.json", ~pd.Series(same), max_rate=0.0,
                           category="leakage")
        suite.rate_at_most("découpage figé: segment new limité aux sources non figées",
                           (r["split"] == "new") & r["source"].isin(spec["sources"]), max_rate=0.0, category="leakage")


def validate_processed(params: dict[str, Any], movies: pd.DataFrame, ratings: pd.DataFrame) -> Suite:
    suite = Suite("processed")
    check_schema(suite, movies, "movies", movies_spec(params))
    check_schema(suite, ratings, "ratings", ratings_spec(params))
    _modality_checks(suite, movies, params)
    _quality_checks(suite, movies, ratings, params)
    _inter_modal_checks(suite, movies, ratings, params)
    _temporal_checks(suite, movies, ratings, params)
    _leakage_checks(suite, movies, ratings, params)
    return suite


def stage(ctx: StageContext) -> None:
    movies = pd.read_parquet(ctx.processed / "movies.parquet")
    ratings = pd.read_parquet(ctx.processed / "ratings.parquet")
    suite = validate_processed(ctx.params, movies, ratings)
    suite.write(ctx.reports / "validation" / "processed.json")
    suite.enforce(ctx)


if __name__ == "__main__":
    main_for("validate_processed", stage, "Contrôles de qualité des tables finales")
