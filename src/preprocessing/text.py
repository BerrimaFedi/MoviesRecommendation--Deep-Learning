"""Étape `preprocess_text` : modalité texte (synopsis + accroche).

Sortie : interim/text.parquet, une ligne par tmdb_id du catalogue — overview, overview_source
(tmdb | kaggle), tagline, title_tmdb, overview_n_chars, overview_n_words, has_overview.
Nettoyage : Unicode NFC, suppression des caractères de contrôle, espaces normalisés, troncature.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from ..common.config import Settings
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from .sources import catalog_keys, coalesce, kaggle_enabled, kaggle_frame, restrict, tmdb_frame

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")


def normalize_text(value: Any, form: str = "NFC", max_len: int | None = None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return None
    s = unicodedata.normalize(form, str(value))
    s = _CONTROL.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0]
    return s or None


def build_text(params: dict[str, Any], catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = Settings.from_params(params)
    tp = params["preprocessing"]["text"]
    t, _ = tmdb_frame(cfg)
    k = kaggle_frame(cfg, kaggle_enabled(params, cfg))
    df = restrict(t, catalog_keys(catalog)).merge(k, on="tmdb_id", how="left")
    norm = lambda s: s.map(lambda v: normalize_text(v, tp["unicode_form"], tp["max_len_chars"]))  # noqa: E731
    df["t_overview"], df["k_overview"] = norm(df["t_overview"]), norm(df["k_overview"])
    # un synopsis trop court (« TBD », titre répété) ne compte pas : on tente alors le secours Kaggle
    short = lambda s: s.where(s.fillna("").str.len() >= tp["min_len_chars"])  # noqa: E731
    df["overview"], df["overview_source"] = coalesce(short(df["t_overview"]), short(df["k_overview"]))
    raw_any, _ = coalesce(df["t_overview"], df["k_overview"])
    df["overview"] = df["overview"].where(df["overview"].notna(), raw_any)  # gardé mais marqué absent
    df.loc[df["overview"].isna(), "overview_source"] = None
    df["tagline"], _ = coalesce(norm(df["t_tagline"]), norm(df["k_tagline"]))
    df["title_tmdb"] = df["t_title"]
    df["overview_n_chars"] = df["overview"].fillna("").str.len().astype("int32")
    df["overview_n_words"] = df["overview"].fillna("").str.split().map(len).astype("int32")
    df["has_overview"] = (df["overview_n_chars"] >= tp["min_len_chars"]).astype("int8")
    out = df[["tmdb_id", "overview", "overview_source", "tagline", "title_tmdb", "overview_n_chars",
              "overview_n_words", "has_overview"]]
    info = {
        "rows": len(out), "has_overview_rate": float(out["has_overview"].mean()) if len(out) else None,
        "source_counts": out["overview_source"].value_counts(dropna=False).to_dict(),
        "rejected_too_short": int(((out["overview_n_chars"] > 0) & (out["has_overview"] == 0)).sum()),
        "words_mean": float(out.loc[out["has_overview"] == 1, "overview_n_words"].mean()) if len(out) else None,
        "words_p95": float(out.loc[out["has_overview"] == 1, "overview_n_words"].quantile(0.95))
        if out["has_overview"].any() else None,
    }
    return out, info


def stage(ctx: StageContext) -> None:
    catalog = pd.read_parquet(ctx.interim / "catalog.parquet")
    out, info = build_text(ctx.params, catalog)
    write_parquet(out, ctx.interim / "text.parquet")
    for k, v in info.items():
        ctx.metric(k, v)


if __name__ == "__main__":
    main_for("preprocess_text", stage, "Prépare la modalité texte")
