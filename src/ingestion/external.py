"""Étape `ingest_external` : données apportées par un tiers (enseignant, binôme) dans `paths.inbox`.

Format attendu (tous les fichiers sont facultatifs ; un dossier vide = aucune donnée externe) :

  inbox/movies.csv    movie_id, title, year, genres, overview, original_language, runtime_min, poster_file
                      - movie_id : entier NOUVEAU (absent de MovieLens) ; genres séparés par « | »
                        (noms MovieLens « Sci-Fi » ou canoniques « Science Fiction ») ; poster_file : nom
                        d'un fichier de inbox/posters/ ; seuls movie_id et title sont obligatoires
  inbox/ratings.csv   user_id, movie_id, rating, timestamp
                      - movie_id : film MovieLens OU film de movies.csv ; user_id : existant ou nouveau ;
                        rating ∈ [0,5 ; 5] au pas de 0,5 ; timestamp : secondes Unix ou date ISO
  inbox/posters/*.jpg affiches (JPEG/PNG), ré-encodées en JPEG RGB

Les contrôles (reports/validation/external.json) sont bloquants : une note à 7★, un film inconnu, un
identifiant en collision ou une affiche illisible arrêtent le pipeline avant tout prétraitement.

Sorties : interim/external/{movies.parquet, ratings.parquet, posters/<movie_id>.jpg}.

  python -m src.ingestion.external --example       # dépose un petit jeu d'exemple valide dans l'inbox
  python -m src.ingestion.external --example-bad   # … ou un jeu volontairement défectueux (démo d'échec)
  python -m src.ingestion.external --clear         # vide l'inbox (CSV et affiches)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..common.config import Settings, load_params
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from ..preprocessing.normalize import CANON_GENRES, ML_GENRE_MAP, normalize_language
from ..preprocessing.text import normalize_text
from ..validation.checks import Suite
from . import movielens

MOVIE_COLS = ["movie_id", "title", "year", "genres", "overview", "original_language", "runtime_min", "poster_file"]
RATING_COLS = ["user_id", "movie_id", "rating", "timestamp"]
SOURCE = "external"


def parse_genres(value: Any, unknown: dict[str, int]) -> list[str]:
    out: list[str] = []
    for raw in str(value).split("|") if isinstance(value, str) else []:
        g = raw.strip()
        if not g:
            continue
        canon = ML_GENRE_MAP.get(g, g if g in CANON_GENRES else None)
        if canon is None and g not in ML_GENRE_MAP:
            unknown[g] = unknown.get(g, 0) + 1
        elif canon and canon not in out:
            out.append(canon)
    order = {g: i for i, g in enumerate(CANON_GENRES)}
    return sorted(out, key=order.__getitem__)


def parse_timestamp(s: pd.Series) -> pd.Series:
    """Secondes Unix (entier) ou date ISO -> secondes Unix ; NaN si illisible."""
    num = pd.to_numeric(s, errors="coerce")
    iso = pd.to_datetime(s.where(num.isna()), errors="coerce", utc=True, format="mixed")
    iso_s = (iso - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return num.where(num.notna(), iso_s)


def _read_csv(path: Path, suite: Suite, name: str, required: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False, na_values=[""])
    except UnicodeDecodeError as e:
        suite.check(f"{name}: encodage UTF-8", False, category="schema", observed=str(e)[:120], threshold="utf-8")
        return None
    suite.check(f"{name}: encodage UTF-8", True, category="schema")
    missing = [c for c in required if c not in df.columns]
    suite.check(f"{name}: colonnes obligatoires", not missing, category="schema", observed=missing or "toutes",
                threshold=required)
    return None if missing else df


def build_external(params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Suite, dict]:
    cfg = Settings.from_params(params)
    inbox = Path(params["paths"]["inbox"])
    out_dir = Path(params["paths"]["interim"]) / "external"
    rp = params["preprocessing"]["ratings"]
    tp = params["preprocessing"]["text"]
    min_side = params["preprocessing"]["image"]["min_side"]
    suite = Suite("external")
    info: dict[str, Any] = {"inbox": inbox.as_posix()}

    ml_movies, _, ml_ratings = movielens.load(cfg.movielens_dir)
    ml_ids = set(ml_movies["movieId"])

    # --- films -----------------------------------------------------------------------------------------
    mv = _read_csv(inbox / "movies.csv", suite, "movies.csv", ["movie_id", "title"])
    movies = pd.DataFrame(columns=["movie_id", "title", "year", "genres", "overview", "original_language",
                                   "runtime_min", "poster_file", "poster_w", "poster_h", "poster_sha1"])
    posters_out = out_dir / "posters"
    if posters_out.exists():
        shutil.rmtree(posters_out)
    posters_out.mkdir(parents=True, exist_ok=True)
    if mv is not None:
        for c in MOVIE_COLS:
            if c not in mv.columns:
                mv[c] = None
        mid = pd.to_numeric(mv["movie_id"], errors="coerce")
        suite.rate_at_most("movies.csv: movie_id entier ≥ 1", ~(mid >= 1), max_rate=0.0, category="schema")
        suite.rate_at_most("movies.csv: movie_id unique", mid.dropna().duplicated(), max_rate=0.0, category="duplicates")
        suite.rate_at_most("movies.csv: movie_id absent de MovieLens (pas de collision)", mid.isin(ml_ids), max_rate=0.0,
                           category="inter_modal", detail=f"collisions : {sorted(mid[mid.isin(ml_ids)].astype(int))[:5]}")
        suite.rate_at_most("movies.csv: titre renseigné", mv["title"].isna(), max_rate=0.0, category="missing")
        year = pd.to_numeric(mv["year"], errors="coerce")
        ymax = datetime.now(UTC).year + params["validation"]["processed"]["year_max_offset"]
        suite.rate_at_most(f"movies.csv: année ∈ [{params['validation']['processed']['year_min']}, {ymax}]",
                           mv["year"].notna() & ~year.between(params["validation"]["processed"]["year_min"], ymax),
                           max_rate=0.0, category="outliers")
        runtime = pd.to_numeric(mv["runtime_min"], errors="coerce")
        mp = params["preprocessing"]["meta"]
        suite.rate_at_most(f"movies.csv: durée ∈ [{mp['runtime_min']}, {mp['runtime_max']}] min",
                           mv["runtime_min"].notna() & ~runtime.between(mp["runtime_min"], mp["runtime_max"]),
                           max_rate=0.0, category="outliers")
        unknown: dict[str, int] = {}
        genres = mv["genres"].map(lambda v: parse_genres(v, unknown))
        suite.check("movies.csv: genres reconnus", not unknown, category="schema", severity="warn",
                    observed=unknown or "tous", threshold=CANON_GENRES, detail="les genres inconnus sont ignorés")
        lang = mv["original_language"].map(lambda v: normalize_language(v) if isinstance(v, str) else "und")
        suite.rate_at_most("movies.csv: code langue ISO 639", ~lang.str.fullmatch(r"[a-z]{2,3}"), max_rate=0.0,
                           category="schema")
        overview = mv["overview"].map(lambda v: normalize_text(v, tp["unicode_form"], tp["max_len_chars"]))

        # affiches : fichier présent, décodable, ré-encodé en JPEG RGB sous interim/external/posters/<id>.jpg
        w, h, sha, files, bad = [], [], [], [], []
        for m, f in zip(mid, mv["poster_file"]):
            if not isinstance(f, str) or not f.strip() or pd.isna(m):
                w.append(np.nan), h.append(np.nan), sha.append(None), files.append(None)
                continue
            src = inbox / "posters" / f.strip()
            try:
                data = src.read_bytes()
                with Image.open(io.BytesIO(data)) as im:
                    rgb = im.convert("RGB")
                if min(rgb.size) < min_side:
                    raise ValueError(f"image trop petite {rgb.size}")
                rgb.save(posters_out / f"{int(m)}.jpg", format="JPEG", quality=90)
                w.append(float(rgb.size[0])), h.append(float(rgb.size[1]))
                sha.append(hashlib.sha1(data).hexdigest()), files.append(f"external/posters/{int(m)}.jpg")
            except (OSError, ValueError) as e:
                bad.append(f"{f}: {str(e)[:60]}")
                w.append(np.nan), h.append(np.nan), sha.append(None), files.append(None)
        n_declared = int(mv["poster_file"].notna().sum())
        suite.rate_at_most("posters/: affiches déclarées présentes et décodables", len(bad), n_declared, max_rate=0.0,
                           category="inter_modal", detail="; ".join(bad[:3]))
        movies = pd.DataFrame({
            "movie_id": mid.astype("Int64"), "title": mv["title"].map(normalize_text), "year": year.astype("Int64"),
            "genres": genres, "overview": overview, "original_language": lang, "runtime_min": runtime,
            "poster_file": files, "poster_w": w, "poster_h": h, "poster_sha1": sha,
        }).dropna(subset=["movie_id"]).drop_duplicates("movie_id")
        movies = movies[~movies["movie_id"].isin(ml_ids)].reset_index(drop=True)

    # --- notes --------------------------------------------------------------------------------------------
    rt = _read_csv(inbox / "ratings.csv", suite, "ratings.csv", RATING_COLS)
    ratings = pd.DataFrame({"user_id": pd.Series(dtype="int32"), "movie_id": pd.Series(dtype="int32"),
                            "rating": pd.Series(dtype="float32"), "timestamp": pd.Series(dtype="int64")})
    if rt is not None:
        uid, mid_r = pd.to_numeric(rt["user_id"], errors="coerce"), pd.to_numeric(rt["movie_id"], errors="coerce")
        rating, ts = pd.to_numeric(rt["rating"], errors="coerce"), parse_timestamp(rt["timestamp"])
        known = set(ml_ids) | set(movies["movie_id"].astype(int))
        on_grid = np.isclose((rating - rp["min"]) / rp["step"], np.round((rating - rp["min"]) / rp["step"]))
        now = datetime.now(UTC).timestamp()
        checks = {
            "ratings.csv: user_id entier ≥ 1": ~(uid >= 1),
            "ratings.csv: movie_id connu (MovieLens ou movies.csv)": ~mid_r.isin(known),
            f"ratings.csv: note ∈ [{rp['min']}, {rp['max']}] au pas de {rp['step']}":
                ~(rating.between(rp["min"], rp["max"]) & on_grid),
            "ratings.csv: horodatage lisible et non futur": ~ts.between(0, now),
        }
        for name, bad_mask in checks.items():
            cat = "inter_modal" if "connu" in name else "schema"
            suite.rate_at_most(name, bad_mask, max_rate=0.0, category=cat)
        dup = pd.DataFrame({"u": uid, "m": mid_r}).duplicated(keep="last")
        suite.rate_at_most("ratings.csv: doublons (user_id, movie_id)", dup, max_rate=0.0, category="duplicates",
                           severity="warn", detail="la dernière ligne est gardée")
        overlap = pd.Series(pd.MultiIndex.from_arrays([uid, mid_r]).isin(
            pd.MultiIndex.from_arrays([ml_ratings["userId"], ml_ratings["movieId"]])), index=rt.index)
        suite.rate_at_most("ratings.csv: paire (user, film) déjà notée dans MovieLens", overlap, max_rate=0.0,
                           category="duplicates", severity="warn", detail="la note externe remplace l'ancienne")
        ok = ~dup & ~pd.concat(list(checks.values()), axis=1).any(axis=1)
        ratings = pd.DataFrame({"user_id": uid[ok].astype("int32"), "movie_id": mid_r[ok].astype("int32"),
                                "rating": rating[ok].astype("float32"), "timestamp": ts[ok].astype("int64")})
    info.update(movies=len(movies), ratings=len(ratings), users=int(ratings["user_id"].nunique()),
                new_users=int(ratings.loc[~ratings["user_id"].isin(ml_ratings["userId"]), "user_id"].nunique()),
                posters=int(movies["poster_file"].notna().sum()) if len(movies) else 0,
                present=bool(len(movies) or len(ratings)))
    return movies, ratings.reset_index(drop=True), suite, info


def stage(ctx: StageContext) -> None:
    movies, ratings, suite, info = build_external(ctx.params)
    out = ctx.interim / "external"
    write_parquet(movies, out / "movies.parquet")
    write_parquet(ratings, out / "ratings.parquet")
    suite.write(ctx.reports / "validation" / "external.json")
    for k, v in info.items():
        ctx.metric(k, v)
    if not info["present"]:
        ctx.info("inbox vide : aucune donnée externe")
    suite.enforce(ctx)


# --- jeux d'exemple pour la démonstration --------------------------------------------------------------------
def _poster(color: tuple[int, int, int]) -> bytes:
    rng = np.random.default_rng(sum(color))
    arr = np.clip(np.array(color) + rng.normal(0, 20, (513, 342, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def write_example(inbox: Path, bad: bool = False) -> None:
    if any(inbox.glob("*.csv")):
        raise SystemExit(f"{inbox} contient déjà des CSV : lancez d'abord --clear")
    (inbox / "posters").mkdir(parents=True, exist_ok=True)
    (inbox / "posters" / "nebula.jpg").write_bytes(_poster((30, 60, 160)))
    (inbox / "posters" / "harvest.jpg").write_bytes(_poster((200, 150, 40)))
    movies = pd.DataFrame([
        {"movie_id": 900001, "title": "Nebula Drift", "year": 2024, "genres": "Sci-Fi|Adventure",
         "overview": "A salvage crew drifting near a dying star finds a derelict ship whose log predicts their future.",
         "original_language": "en", "runtime_min": 118, "poster_file": "nebula.jpg"},
        {"movie_id": 900002, "title": "La Moisson", "year": 2023, "genres": "Drama|Romance",
         "overview": "Dans un village viticole, deux familles rivales doivent unir leurs forces le temps d'une vendange.",
         "original_language": "fr", "runtime_min": 104, "poster_file": "harvest.jpg"},
    ])
    ts = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp())
    ratings = pd.DataFrame([
        {"user_id": 1, "movie_id": 900001, "rating": 4.5, "timestamp": ts},
        {"user_id": 7, "movie_id": 900001, "rating": 4.0, "timestamp": ts + 3600},
        {"user_id": 7, "movie_id": 900002, "rating": 3.5, "timestamp": ts + 7200},
        {"user_id": 9001, "movie_id": 900002, "rating": 5.0, "timestamp": "2025-06-02T10:00:00Z"},
        {"user_id": 9001, "movie_id": 1, "rating": 4.0, "timestamp": "2025-06-02T10:05:00Z"},
    ])
    if bad:
        movies.loc[1, "movie_id"] = 1                       # collision avec « Toy Story » de MovieLens
        movies.loc[0, "poster_file"] = "missing.jpg"        # affiche introuvable
        ratings.loc[0, "rating"] = 7.0                      # note hors domaine
        ratings.loc[1, "movie_id"] = 123456789              # film inconnu
    movies.to_csv(inbox / "movies.csv", index=False)
    ratings.to_csv(inbox / "ratings.csv", index=False)
    print(f"jeu {'DÉFECTUEUX' if bad else 'valide'} déposé dans {inbox}/ : {len(movies)} films, {len(ratings)} notes")


def clear(inbox: Path) -> None:
    for f in inbox.glob("*.csv"):
        f.unlink()
    shutil.rmtree(inbox / "posters", ignore_errors=True)
    print(f"{inbox}/ vidé (README conservé)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--example", action="store_true")
    g.add_argument("--example-bad", action="store_true")
    g.add_argument("--clear", action="store_true")
    a, _ = ap.parse_known_args()
    inbox = Path(load_params(a.params, a.overrides)["paths"]["inbox"])
    if a.example or a.example_bad:
        write_example(inbox, bad=a.example_bad)
    elif a.clear:
        clear(inbox)
    else:
        main_for("ingest_external", stage, "Valide et normalise les données externes de l'inbox")


if __name__ == "__main__":
    main()
