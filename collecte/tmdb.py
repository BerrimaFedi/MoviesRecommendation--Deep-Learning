"""Collecte asynchrone des métadonnées TMDB : débit limité, reprises, cache JSONL reprenable.

Principes :
- un enregistrement par film dans data/raw/tmdb/movies.jsonl : {tmdb_id, status, fetched_at, data}
- seuls les statuts 200 (trouvé) et 404 (inconnu de TMDB) sont persistés ; les erreurs temporaires
  (429, 5xx, réseau) sont retentées avec attente exponentielle, puis laissées à la prochaine exécution ;
- relancer la commande ne refait que ce qui manque ou a dépassé la durée de cache autorisée.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp

from .config import Settings


def _ssl_context() -> ssl.SSLContext | bool:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        return ssl.create_default_context()
    except Exception:
        return False


class AuthError(RuntimeError):
    pass


class RateLimiter:
    """Espace les requêtes d'au moins 1/rate seconde (partagé entre toutes les tâches)."""

    def __init__(self, rate_per_sec: float) -> None:
        self.interval = 1.0 / rate_per_sec
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            t = max(time.monotonic(), self._next)
            self._next = t + self.interval
        delay = t - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    """Dernier enregistrement par identifiant. Ignore une dernière ligne tronquée (arrêt brutal)."""
    cache: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[int(rec["tmdb_id"])] = rec
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return cache


def select_todo(ids: list[int], cache: dict[int, dict], cfg: Settings, retry_404: bool = False) -> list[int]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.cache_max_age_days)
    todo = []
    for i in ids:
        rec = cache.get(i)
        if rec is None:
            todo.append(i)
            continue
        if rec.get("status") == 404 and not retry_404:
            continue
        try:
            fetched = datetime.fromisoformat(rec["fetched_at"])
        except (KeyError, ValueError):
            todo.append(i)
            continue
        if fetched < cutoff:  # cache périmé (limite de conservation des conditions d'API)
            todo.append(i)
    return todo


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(120.0, float(retry_after))
        except ValueError:
            pass
    return min(60.0, 2.0 ** attempt + random.random())


async def fetch_one(session: aiohttp.ClientSession, tmdb_id: int, cfg: Settings,
                    limiter: RateLimiter, params: dict[str, str]) -> dict[str, Any]:
    url = f"{cfg.tmdb_base_url}/movie/{tmdb_id}"
    q = {**params, "language": cfg.language}
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
    last = "max_retries"
    for attempt in range(cfg.max_retries + 1):
        await limiter.wait()
        try:
            async with session.get(url, params=q, timeout=timeout) as r:
                if r.status == 200:
                    return {"tmdb_id": tmdb_id, "status": 200, "fetched_at": utcnow_iso(),
                            "data": await r.json()}
                if r.status == 404:
                    return {"tmdb_id": tmdb_id, "status": 404, "fetched_at": utcnow_iso(), "data": None}
                if r.status in (401, 403):
                    raise AuthError(f"TMDB a refusé les identifiants (HTTP {r.status}).")
                if r.status == 429 or r.status >= 500:
                    last = f"http_{r.status}"
                    await asyncio.sleep(_backoff(attempt, r.headers.get("Retry-After")))
                    continue
                return {"tmdb_id": tmdb_id, "status": r.status, "fetched_at": utcnow_iso(), "data": None}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = type(e).__name__
            await asyncio.sleep(_backoff(attempt, None))
    return {"tmdb_id": tmdb_id, "status": -1, "fetched_at": utcnow_iso(), "data": None, "error": last}


async def fetch_configuration(cfg: Settings, params: dict[str, str], headers: dict[str, str]) -> dict | None:
    """/configuration : adresse de base des images et tailles d'affiche disponibles."""
    connector = aiohttp.TCPConnector(ssl=_ssl_context())
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        try:
            async with session.get(f"{cfg.tmdb_base_url}/configuration", params=params,
                                   timeout=aiohttp.ClientTimeout(total=cfg.request_timeout_s)) as r:
                if r.status in (401, 403):
                    raise AuthError(f"TMDB a refusé les identifiants (HTTP {r.status}).")
                if r.status != 200:
                    return None
                conf = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
    cfg.tmdb_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmdb_config_file.write_text(json.dumps(conf, indent=2), encoding="utf-8")
    sizes = conf.get("images", {}).get("poster_sizes", [])
    if sizes and cfg.poster_size not in sizes:
        raise ValueError(f"Taille d'affiche '{cfg.poster_size}' indisponible. Tailles TMDB : {sizes}")
    return conf


def compact_cache(path: Path) -> None:
    """Réécrit le JSONL en gardant une seule ligne par film."""
    cache = load_cache(path)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in cache.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


async def _run(cfg: Settings, ids: list[int], retry_404: bool) -> Counter:
    headers, params = cfg.tmdb_auth()
    cfg.tmdb_dir.mkdir(parents=True, exist_ok=True)
    await fetch_configuration(cfg, params, headers)

    cache = load_cache(cfg.tmdb_cache)
    todo = select_todo(ids, cache, cfg, retry_404)
    stats: Counter = Counter()
    stats["demandes"] = len(ids)
    stats["deja_en_cache"] = len(ids) - len(todo)
    if not todo:
        return stats

    limiter = RateLimiter(cfg.rate_per_sec)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in todo:
        queue.put_nowait(i)
    stop = asyncio.Event()
    failure: list[Exception] = []
    done = 0
    total = len(todo)

    connector = aiohttp.TCPConnector(limit=cfg.concurrency, ssl=_ssl_context())
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        with cfg.tmdb_cache.open("a", encoding="utf-8") as out:
            async def worker() -> None:
                nonlocal done
                while not stop.is_set():
                    try:
                        tid = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        rec = await fetch_one(session, tid, cfg, limiter, params)
                    except AuthError as e:
                        failure.append(e)
                        stop.set()
                        return
                    if rec["status"] in (200, 404):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        stats["ok" if rec["status"] == 200 else "introuvable_404"] += 1
                    else:
                        stats["echec_a_retenter"] += 1
                    done += 1
                    if done % 500 == 0:
                        print(f"[tmdb] {done}/{total}")

            await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))

    if failure:
        raise failure[0]
    compact_cache(cfg.tmdb_cache)
    return stats


def fetch(cfg: Settings, ids: list[int], retry_404: bool = False) -> Counter:
    ids = sorted(set(int(i) for i in ids))
    stats = asyncio.run(_run(cfg, ids, retry_404))
    print(f"[tmdb] terminé : {dict(stats)}")
    return stats


def cache_to_records(path: Path) -> list[dict[str, Any]]:
    """Liste aplatie des films trouvés (status 200) avec la date de collecte."""
    out = []
    for tid, rec in load_cache(path).items():
        if rec.get("status") == 200 and isinstance(rec.get("data"), dict):
            d = dict(rec["data"])
            d["_tmdb_id"] = tid
            d["_fetched_at"] = rec["fetched_at"]
            out.append(d)
    return out
