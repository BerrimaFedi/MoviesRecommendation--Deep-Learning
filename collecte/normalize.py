"""Normalisation des métadonnées : genres, langues, textes, années.

C'est ici que se règle le point de vigilance DataOps « cohérence des métadonnées entre sources » :
MovieLens, TMDB et Kaggle n'utilisent ni les mêmes noms de genres ni les mêmes conventions.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable

# Vocabulaire canonique (= genres TMDB « film » + Film-Noir, qui n'existe que chez MovieLens).
CANON_GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family",
    "Fantasy", "Film-Noir", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]

# MovieLens -> canonique. IMAX est un format de projection, pas un genre : traité à part.
ML_GENRE_MAP: dict[str, str | None] = {
    "Action": "Action", "Adventure": "Adventure", "Animation": "Animation",
    "Children": "Family", "Comedy": "Comedy", "Crime": "Crime", "Documentary": "Documentary",
    "Drama": "Drama", "Fantasy": "Fantasy", "Film-Noir": "Film-Noir", "Horror": "Horror",
    "Musical": "Music", "Mystery": "Mystery", "Romance": "Romance", "Sci-Fi": "Science Fiction",
    "Thriller": "Thriller", "War": "War", "Western": "Western",
    "IMAX": None, "(no genres listed)": None,
}

# TMDB / Kaggle -> canonique. « Foreign » (ancien genre TMDB supprimé) est ignoré.
TMDB_GENRE_MAP: dict[str, str | None] = {g: g for g in CANON_GENRES if g != "Film-Noir"}
TMDB_GENRE_MAP["Foreign"] = None

LANG_ALIASES = {"cn": "zh", "iw": "he", "in": "id", "ji": "yi", "jw": "jv", "mo": "ro", "xx": "und"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def genre_columns() -> list[str]:
    return [f"g_{slug(g)}" for g in CANON_GENRES]


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_ml_genres(value: Any, unmapped: dict[str, int] | None = None) -> tuple[list[str], bool]:
    """'Adventure|Children|Sci-Fi' -> (['Adventure','Family','Science Fiction'], is_imax)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [], False
    out: list[str] = []
    imax = False
    for raw in str(value).split("|"):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "IMAX":
            imax = True
        if raw in ML_GENRE_MAP:
            if ML_GENRE_MAP[raw]:
                out.append(ML_GENRE_MAP[raw])  # type: ignore[arg-type]
        elif unmapped is not None:
            unmapped[raw] = unmapped.get(raw, 0) + 1
    return _dedupe(out), imax


def parse_tmdb_genres(value: Any, unmapped: dict[str, int] | None = None) -> list[str]:
    """Accepte une liste de dicts {'id','name'}, une liste de noms, ou une chaîne Kaggle
    (littéral Python stringifié, ex. "[{'id': 28, 'name': 'Action'}]")."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    names: list[str] = []
    for item in value if isinstance(value, (list, tuple)) else []:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str):
            continue
        if name in TMDB_GENRE_MAP:
            if TMDB_GENRE_MAP[name]:
                names.append(TMDB_GENRE_MAP[name])  # type: ignore[arg-type]
        elif unmapped is not None:
            unmapped[name] = unmapped.get(name, 0) + 1
    return _dedupe(names)


def jaccard(a: list[str], b: list[str]) -> float | None:
    if not a or not b:
        return None
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def normalize_language(code: Any) -> str:
    """Code ISO 639-1 en minuscules ; 'und' si inconnu ou sans contenu linguistique."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "und"
    c = str(code).strip().lower().replace("_", "-").split("-")[0]
    if not c:
        return "und"
    return LANG_ALIASES.get(c, c)


_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def split_title_year(title: Any) -> tuple[str | None, int | None]:
    """'Toy Story (1995)' -> ('Toy Story', 1995). Le titre MovieLens garde sa forme sinon."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return None, None
    t = str(title).strip()
    m = _YEAR_RE.search(t)
    if not m:
        return t, None
    return t[: m.start()].strip(), int(m.group(1))


def clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s or None


def to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def imdb_tt(imdb_id: Any) -> str | None:
    """MovieLens stocke l'identifiant IMDb sans zéros de tête : 114709 -> 'tt0114709'."""
    try:
        return f"tt{int(imdb_id):07d}"
    except (TypeError, ValueError):
        return None
