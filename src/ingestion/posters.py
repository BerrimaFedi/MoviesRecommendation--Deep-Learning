"""Téléchargement et validation des affiches (Pillow). Reprenable, sans écraser ce qui est valide."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

from ..common.config import Settings
from .tmdb import RateLimiter, _ssl_context, cache_to_records, utcnow_iso


def collect_poster_paths(cfg: Settings, wanted: set[int] | None = None) -> dict[int, str]:
    """poster_path par tmdb_id : TMDB d'abord, Kaggle (instantané 2017) en secours.

    `wanted` restreint aux tmdbId du catalogue MovieLens. Correction de la Tâche 2 : sans ce filtre,
    le secours Kaggle ajoutait les ~45 000 films Kaggle, dont la plupart sont hors catalogue.
    """
    paths: dict[int, str] = {}
    for rec in cache_to_records(cfg.tmdb_cache):
        p = rec.get("poster_path")
        if isinstance(p, str) and p.startswith("/"):
            paths[rec["_tmdb_id"]] = p
    try:
        from . import kaggle
        kdf = kaggle.load(cfg)
    except Exception as e:  # noqa: BLE001 - le dataset Kaggle est facultatif
        print(f"[affiches] Kaggle indisponible ({e}) : secours ignoré")
        kdf = None
    if kdf is not None:
        for tid, p in zip(kdf["tmdb_id"], kdf["poster_path"]):
            if isinstance(p, str) and p.startswith("/") and int(tid) not in paths:
                paths[int(tid)] = p
    if wanted is not None:
        paths = {t: p for t, p in paths.items() if t in wanted}
    return paths


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[int(rec["tmdb_id"])] = rec
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def poster_file(cfg: Settings, tmdb_id: int) -> Path:
    return cfg.posters_dir / f"{tmdb_id}.jpg"


def _image_base(cfg: Settings) -> str:
    if cfg.tmdb_config_file.exists():
        try:
            conf = json.loads(cfg.tmdb_config_file.read_text(encoding="utf-8"))
            base = conf.get("images", {}).get("secure_base_url")
            if base:
                return base
        except json.JSONDecodeError:
            pass
    return cfg.tmdb_image_base


def validate_and_save(data: bytes, dest: Path, min_side: int = 16) -> tuple[int, int, str]:
    """Vérifie que les octets sont une vraie image, l'enregistre en JPEG RGB, retourne (l, h, sha1)."""
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    if min(w, h) < min_side:
        raise ValueError(f"image trop petite : {w}x{h}")
    tmp = dest.with_suffix(".tmp")
    rgb.save(tmp, format="JPEG", quality=90)
    os.replace(tmp, dest)
    return w, h, hashlib.sha1(data).hexdigest()


def _needs_work(cfg: Settings, tid: int, path: str, manifest: dict, retry_failed: bool) -> bool:
    rec = manifest.get(tid)
    if rec is None or rec.get("poster_path") != path:
        return True
    if rec["status"] == "ok":
        return not poster_file(cfg, tid).exists()
    return retry_failed  # http_404 / invalid_image : définitifs sauf demande explicite


async def _download_one(session: aiohttp.ClientSession, cfg: Settings, limiter: RateLimiter,
                        base: str, tid: int, path: str) -> dict[str, Any]:
    url = f"{base}{cfg.poster_size}{path}"
    rec: dict[str, Any] = {"tmdb_id": tid, "poster_path": path, "fetched_at": utcnow_iso()}
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
    for attempt in range(cfg.max_retries + 1):
        await limiter.wait()
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    data = await r.read()
                    try:
                        w, h, sha = await asyncio.to_thread(validate_and_save, data, poster_file(cfg, tid),
                                                         cfg.image_min_side)
                    except Exception as e:  # noqa: BLE001  # image corrompue, HTML d'erreur, etc.
                        return {**rec, "status": "invalid_image", "detail": str(e)[:200]}
                    return {**rec, "status": "ok", "width": w, "height": h, "bytes": len(data), "sha1": sha}
                if r.status == 404:
                    return {**rec, "status": "http_404"}
                if r.status == 429 or r.status >= 500:
                    ra = r.headers.get("Retry-After")
                    await asyncio.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit()
                                        else min(60.0, 2.0 ** attempt + random.random()))
                    continue
                return {**rec, "status": f"http_{r.status}"}
        except (TimeoutError, aiohttp.ClientError):
            await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
    return {**rec, "status": "transient_error"}


async def _run(cfg: Settings, items: dict[int, str], retry_failed: bool) -> Counter:
    cfg.posters_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg.posters_manifest)
    todo = [(t, p) for t, p in sorted(items.items()) if _needs_work(cfg, t, p, manifest, retry_failed)]
    stats: Counter = Counter()
    stats["demandees"] = len(items)
    stats["deja_ok"] = len(items) - len(todo)
    if not todo:
        return stats

    base = _image_base(cfg)
    limiter = RateLimiter(cfg.rate_per_sec * 2)  # le CDN d'images n'est pas soumis à la limite de l'API
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for it in todo:
        queue.put_nowait(it)
    done = 0

    connector = aiohttp.TCPConnector(limit=cfg.concurrency, ssl=_ssl_context())
    async with aiohttp.ClientSession(connector=connector) as session:
        with cfg.posters_manifest.open("a", encoding="utf-8") as out:
            async def worker() -> None:
                nonlocal done
                while True:
                    try:
                        tid, path = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    rec = await _download_one(session, cfg, limiter, base, tid, path)
                    stats[rec["status"]] += 1
                    if rec["status"] != "transient_error":
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                    done += 1
                    if done % 500 == 0:
                        print(f"[affiches] {done}/{len(todo)}")

            await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))

    # compactage : une ligne par film
    final = load_manifest(cfg.posters_manifest)
    tmp = cfg.posters_manifest.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in final.values():
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, cfg.posters_manifest)
    return stats


def fetch(cfg: Settings, retry_failed: bool = False, wanted: set[int] | None = None) -> Counter:
    items = collect_poster_paths(cfg, wanted)
    stats = asyncio.run(_run(cfg, items, retry_failed))
    print(f"[affiches] terminé : {dict(stats)}")
    return stats


def stage(ctx) -> None:
    """Étape `ingest_posters` : télécharge et valide l'affiche de chaque film du catalogue."""
    from . import movielens

    cfg = Settings.from_params(ctx.params)
    if ctx.params["ingestion"]["mode"] == "synthetic":
        from .synthetic import make_posters
        make_posters(ctx.params)
    else:
        _, links, _ = movielens.load(cfg.movielens_dir)
        wanted = {int(i) for i in links["tmdbId"].dropna()}
        stats = fetch(cfg, wanted=wanted)
        for k, v in stats.items():
            ctx.metric(f"fetch_{k}", v)
    manifest = load_manifest(cfg.posters_manifest)
    statuses = Counter(r["status"] for r in manifest.values())
    ctx.metric("manifest_records", len(manifest))
    ctx.metric("manifest_status", dict(statuses))


def prune_orphans(cfg: Settings, wanted: set[int]) -> tuple[int, int]:
    """Supprime les affiches et entrées de manifeste hors catalogue (héritage du secours Kaggle non filtré
    de la Tâche 2). Renvoie (fichiers supprimés, octets libérés)."""
    n, size = 0, 0
    for f in cfg.posters_dir.glob("*.jpg"):
        if f.stem.isdigit() and int(f.stem) not in wanted:
            size += f.stat().st_size
            f.unlink()
            n += 1
    manifest = load_manifest(cfg.posters_manifest)
    tmp = cfg.posters_manifest.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for tid, rec in manifest.items():
            if tid in wanted:
                out.write(json.dumps(rec) + "\n")
    os.replace(tmp, cfg.posters_manifest)
    return n, size


if __name__ == "__main__":
    import sys

    from ..common.config import load_params
    from ..common.stage import main_for

    if "--prune-orphans" in sys.argv:
        from . import movielens

        params = load_params()
        cfg = Settings.from_params(params)
        _, links, _ = movielens.load(cfg.movielens_dir)
        n, size = prune_orphans(cfg, {int(i) for i in links["tmdbId"].dropna()})
        print(f"[affiches] {n} affiches hors catalogue supprimées ({size / 1e6:.1f} Mo)")
    else:
        main_for("ingest_posters", stage, "Télécharge et valide les affiches")
