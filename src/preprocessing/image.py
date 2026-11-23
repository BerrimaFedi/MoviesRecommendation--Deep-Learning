"""Étape `preprocess_images` : modalité image (affiches).

Sortie : interim/images.parquet, une ligne par tmdb_id du catalogue — poster_path, poster_path_source,
poster_file (relatif au dossier parent de paths.raw_posters), poster_w, poster_h, poster_sha1,
poster_decode_ok, has_poster. Le décodage Pillow est vérifié sur tout ou partie des fichiers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..common.config import Settings
from ..common.io import write_parquet
from ..common.stage import StageContext, main_for
from ..ingestion.posters import load_manifest, poster_file
from .sources import catalog_keys, coalesce, kaggle_enabled, kaggle_frame, restrict, tmdb_frame


def decodes(path, min_side: int) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
            return im.mode == "RGB" and min(im.size) >= min_side
    except Exception:  # noqa: BLE001
        return False


def build_images(params: dict[str, Any], catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = Settings.from_params(params)
    ip = params["preprocessing"]["image"]
    t, _ = tmdb_frame(cfg)
    k = kaggle_frame(cfg, kaggle_enabled(params, cfg))
    df = restrict(t[["tmdb_id", "t_poster_path"]], catalog_keys(catalog)).merge(
        k[["tmdb_id", "k_poster_path"]], on="tmdb_id", how="left")
    df["poster_path"], df["poster_path_source"] = coalesce(df["t_poster_path"], df["k_poster_path"])

    manifest = load_manifest(cfg.posters_manifest)
    pm = pd.DataFrame(list(manifest.values()), columns=["tmdb_id", "status", "width", "height", "sha1"])
    pm = pm[pm["status"] == "ok"].rename(columns={"width": "poster_w", "height": "poster_h",
                                                    "sha1": "poster_sha1"}).drop(columns="status")
    pm["tmdb_id"] = pm["tmdb_id"].astype("Int64")
    df = df.merge(pm, on="tmdb_id", how="left")
    exists = np.array([pd.notna(s) and poster_file(cfg, int(i)).exists()
                       for i, s in zip(df["tmdb_id"], df["poster_sha1"])], dtype=bool)

    # vérification du décodage : tout, un échantillon déterministe, ou rien
    mode = ip["verify_decode"]
    decode_ok = pd.Series(pd.NA, index=df.index, dtype="boolean")
    idx = np.flatnonzero(exists)
    if mode == "sample" and len(idx) > ip["verify_sample"]:
        idx = np.random.default_rng(params["project"]["seed"]).choice(idx, ip["verify_sample"], replace=False)
    if mode != "none":
        for i in idx:
            decode_ok.iat[i] = decodes(poster_file(cfg, int(df["tmdb_id"].iat[i])), ip["min_side"])
    df["poster_decode_ok"] = decode_ok
    df["has_poster"] = (exists & decode_ok.fillna(True).to_numpy(dtype=bool)).astype("int8")
    prefix = cfg.posters_dir.name
    df["poster_file"] = [f"{prefix}/{int(i)}.jpg" if h else None for i, h in zip(df["tmdb_id"], df["has_poster"])]
    out = df[["tmdb_id", "poster_path", "poster_path_source", "poster_file", "poster_w", "poster_h", "poster_sha1",
              "poster_decode_ok", "has_poster"]]
    checked = int(decode_ok.notna().sum())
    info = {
        "rows": len(out), "has_poster_rate": float(out["has_poster"].mean()) if len(out) else None,
        "manifest_ok_but_file_missing": int((df["poster_sha1"].notna() & ~exists).sum()),
        "decode_checked": checked, "decode_failed": int((decode_ok == False).sum()),  # noqa: E712
        "manifest_records_total": len(manifest),
        "manifest_records_outside_catalog": len(set(manifest) - set(catalog["tmdb_id"].dropna().astype(int))),
    }
    return out, info


def stage(ctx: StageContext) -> None:
    catalog = pd.read_parquet(ctx.interim / "catalog.parquet")
    out, info = build_images(ctx.params, catalog)
    write_parquet(out, ctx.interim / "images.parquet")
    for k, v in info.items():
        ctx.metric(k, v)
    if info["manifest_records_outside_catalog"]:
        ctx.warn("entrées du manifeste d'affiches hors catalogue MovieLens (ignorées)",
                 n=info["manifest_records_outside_catalog"])


if __name__ == "__main__":
    main_for("preprocess_images", stage, "Prépare la modalité image")
