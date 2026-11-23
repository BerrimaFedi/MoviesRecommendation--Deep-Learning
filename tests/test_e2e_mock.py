"""Test de bout en bout contre un faux serveur TMDB local (aucun accès réseau réel).

Le faux serveur reproduit les cas difficiles : 429 puis succès, 500 puis succès, 404, genre inconnu,
synopsis vide, affiche absente, affiche en 404, affiche corrompue, durée 0, codes de langue atypiques.
"""
import asyncio
import io
import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from aiohttp import web
from PIL import Image

from src.common.config import MissingCredentials, Settings, set_path
from src.common.io import read_json
from src.ingestion import kaggle as kaggle_meta
from src.ingestion import posters, tmdb
from src.pipeline import run
from src.preprocessing.normalize import genre_columns
from tests.conftest import params_for

TOKEN = "test-token"
LONG = "A sufficiently long synopsis written for the mock server, number {}."


class Mock:
    def __init__(self) -> None:
        self.hits: dict[str, int] = {}

    def count(self, key: str) -> int:
        self.hits[key] = self.hits.get(key, 0) + 1
        return self.hits[key]

    @staticmethod
    def jpeg() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (342, 513), (120, 30, 200)).save(buf, "JPEG")
        return buf.getvalue()

    def app(self, port_holder: dict) -> web.Application:
        async def configuration(req: web.Request) -> web.Response:
            if req.headers.get("Authorization") != f"Bearer {TOKEN}":
                return web.Response(status=401)
            return web.json_response({"images": {
                "secure_base_url": f"http://127.0.0.1:{port_holder['port']}/t/p/",
                "poster_sizes": ["w92", "w342", "original"]}})

        async def movie(req: web.Request) -> web.Response:
            if req.headers.get("Authorization") != f"Bearer {TOKEN}":
                return web.Response(status=401)
            tid = int(req.match_info["id"])
            n = self.count(f"movie/{tid}")
            if tid == 99:
                return web.Response(status=404)
            if tid == 11 and n == 1:
                return web.Response(status=429, headers={"Retry-After": "0"})
            if tid == 12 and n == 1:
                return web.Response(status=500)
            genres = ([{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}] if tid % 2 == 0
                      else [{"id": 35, "name": "Comedy"}, {"id": 10751, "name": "Family"}])
            if tid == 13:
                genres = genres + [{"id": 10764, "name": "Reality"}]  # genre hors vocabulaire
            data = {"id": tid, "title": f"Movie {tid}", "overview": "" if tid == 14 else LONG.format(tid),
                    "tagline": "", "genres": genres, "runtime": 0 if tid == 18 else 100,
                    "popularity": 12.5, "vote_average": 7.1, "vote_count": 100,
                    "original_language": {19: "cn", 20: "xx"}.get(tid, "en"),
                    "release_date": f"{1990 + tid % 10}-01-01",
                    "poster_path": None if tid == 15 else f"/p{tid}.jpg", "adult": False}
            return web.json_response(data)

        async def image(req: web.Request) -> web.Response:
            name = req.match_info["name"]
            self.count(f"image/{name}")
            if name == "p16.jpg":
                return web.Response(status=404)
            if name == "p17.jpg":
                return web.Response(body=b"<html>not an image</html>", content_type="text/html")
            return web.Response(body=self.jpeg(), content_type="image/jpeg")

        app = web.Application()
        app.router.add_get("/3/configuration", configuration)
        app.router.add_get("/3/movie/{id}", movie)
        app.router.add_get("/t/p/{size}/{name}", image)
        return app


def start_server(mock: Mock):
    holder: dict = {}
    ready = threading.Event()

    def run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(mock.app(holder))
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        holder.update(port=site._server.sockets[0].getsockname()[1], loop=loop)
        ready.set()
        loop.run_forever()
        loop.run_until_complete(runner.cleanup())

    th = threading.Thread(target=run, daemon=True)
    th.start()
    assert ready.wait(10)
    return holder


def make_movielens(cfg: Settings) -> None:
    d = cfg.movielens_dir
    d.mkdir(parents=True)
    rows, links, ratings = [], [], []
    for mid in range(1, 27):
        tid = mid + 10
        genres = "Action|Sci-Fi|IMAX" if tid % 2 == 0 else "Comedy|Children"
        year = 1990 + tid % 10
        rows.append((mid, f"Movie {tid} ({year})", genres))
        links.append((mid, 100000 + mid, "" if mid == 26 else (99 if mid == 25 else tid)))
    pd.DataFrame(rows, columns=["movieId", "title", "genres"]).to_csv(d / "movies.csv", index=False)
    pd.DataFrame(links, columns=["movieId", "imdbId", "tmdbId"]).to_csv(d / "links.csv", index=False)
    import numpy as np
    rng = np.random.default_rng(0)
    base_ts = 1420070400  # 2015-01-01
    for u in range(1, 31):
        for m in rng.choice(np.arange(1, 27), size=15, replace=False):
            ratings.append((u, int(m), float(rng.integers(1, 11)) / 2,
                            base_ts + int(rng.integers(0, 4 * 365 * 86400))))
    pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(d / "ratings.csv", index=False)


def make_kaggle(cfg: Settings) -> None:
    cfg.kaggle_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "imdb_id", "title", "overview", "tagline", "genres", "runtime", "popularity",
            "vote_average", "vote_count", "original_language", "release_date", "poster_path"]
    g = "[{'id': 35, 'name': 'Comedy'}]"
    rows = [
        ["14", "tt1", "Movie 14", "Fallback synopsis coming from the Kaggle snapshot, long enough.", "", g,
         "90", "5", "6", "10", "en", "1994-01-01", "/k14.jpg"],
        ["15", "tt2", "Movie 15", "Another long Kaggle synopsis for the movie without poster.", "", g,
         "90", "5", "6", "10", "en", "1995-01-01", "/k15.jpg"],
        ["15", "tt2", "Movie 15", "Duplicate row with fewer votes.", "", g, "90", "5", "6", "2", "en",
         "1995-01-01", "/dup.jpg"],
        ["1997-08-20", "", "Shifted row", "", "", "", "", "", "", "", "", "", ""],  # ligne malformée connue
    ]
    pd.DataFrame(rows, columns=cols).to_csv(cfg.kaggle_dir / "movies_metadata.csv", index=False)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    mock = Mock()
    srv = start_server(mock)
    port = srv["port"]
    tmp = tmp_path_factory.mktemp("data")
    cfg = Settings.for_data_dir(tmp, "ml-mock", tmdb_base_url=f"http://127.0.0.1:{port}/3",
                                tmdb_image_base=f"http://127.0.0.1:{port}/t/p/", rate_per_sec=300, concurrency=8,
                                max_retries=3)
    params = params_for(tmp, raw_movielens="ml-mock")
    # jeu minuscule et volontairement défectueux : seuils de volume et de couverture relâchés
    for key, value in {"split.cold_item_frac": 0.1, "validation.raw.min_movies": 10, "validation.raw.min_ratings": 100,
                       "validation.processed.min_all_modalities_rate": 0.5, "validation.processed.max_missing":
                       {"overview": 0.2, "poster_file": 0.3, "release_year": 0.1, "runtime_min": 0.2,
                        "original_language": 0.0}}.items():
        set_path(params, key, value)
    make_movielens(cfg)
    make_kaggle(cfg)
    ns = SimpleNamespace(cfg=cfg, mock=mock, port=port)

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("TMDB_API_KEY", raising=False)
        mp.delenv("TMDB_READ_TOKEN", raising=False)
        with pytest.raises(MissingCredentials):
            tmdb.fetch(cfg, [11])

        mp.setenv("TMDB_READ_TOKEN", "wrong")
        with pytest.raises(tmdb.AuthError):
            tmdb.fetch(cfg, [11])
        assert not cfg.tmdb_cache.exists() or cfg.tmdb_cache.read_text() == ""

        mp.setenv("TMDB_READ_TOKEN", TOKEN)
        links = pd.read_csv(cfg.movielens_dir / "links.csv")
        ids = sorted(int(i) for i in links["tmdbId"].dropna().unique())
        ns.stats1 = tmdb.fetch(cfg, ids)
        ns.hits_after_first = sum(v for k, v in mock.hits.items() if k.startswith("movie/"))
        ns.stats2 = tmdb.fetch(cfg, ids)  # reprise : rien à refaire
        ns.hits_after_second = sum(v for k, v in mock.hits.items() if k.startswith("movie/"))
        ns.pstats1 = posters.fetch(cfg)
        ns.image_hits_1 = sum(v for k, v in mock.hits.items() if k.startswith("image/"))
        ns.pstats2 = posters.fetch(cfg)
        ns.image_hits_2 = sum(v for k, v in mock.hits.items() if k.startswith("image/"))
        ns.exit_code = run(params, start="ingest_external")

    ns.params = params
    reports = tmp / "reports"
    ns.stage = lambda name: read_json(reports / "stages" / f"{name}.json")["metrics"]
    ns.reports = reports
    ns.movies = pd.read_parquet(tmp / "processed" / "movies.parquet")
    ns.ratings = pd.read_parquet(tmp / "processed" / "ratings.parquet")
    yield ns
    srv["loop"].call_soon_threadsafe(srv["loop"].stop)


def row(ns, tmdb_id: int) -> pd.Series:
    return ns.movies[ns.movies["tmdb_id"] == tmdb_id].iloc[0]


def test_tmdb_fetch_retries_and_classifies(pipeline):
    s = pipeline.stats1
    assert s["ok"] == 24 and s["introuvable_404"] == 1 and s["echec_a_retenter"] == 0
    assert pipeline.mock.hits["movie/11"] == 2  # 429 puis succès
    assert pipeline.mock.hits["movie/12"] == 2  # 500 puis succès


def test_tmdb_fetch_is_resumable(pipeline):
    assert pipeline.stats2["deja_en_cache"] == 25
    assert pipeline.hits_after_second == pipeline.hits_after_first


def test_posters_statuses(pipeline):
    s = pipeline.pstats1
    assert s["ok"] == 22 and s["http_404"] == 1 and s["invalid_image"] == 1
    m = posters.load_manifest(pipeline.cfg.posters_manifest)
    assert len(m) == 24
    with Image.open(pipeline.cfg.posters_dir / "11.jpg") as im:
        assert im.size == (342, 513) and im.mode == "RGB"
    assert not (pipeline.cfg.posters_dir / "17.jpg").exists()  # image corrompue non conservée


def test_posters_are_not_downloaded_twice(pipeline):
    assert pipeline.pstats2["deja_ok"] == 24
    assert pipeline.image_hits_2 == pipeline.image_hits_1


def test_kaggle_loader_cleans_malformed_and_duplicate_rows(pipeline):
    k = kaggle_meta.load(pipeline.cfg)
    assert sorted(k["tmdb_id"]) == [14, 15]
    assert k.loc[k["tmdb_id"] == 15, "poster_path"].iloc[0] == "/k15.jpg"  # doublon : plus de votes gardé


def test_kaggle_fallbacks_fill_missing_modalities(pipeline):
    r14, r15 = row(pipeline, 14), row(pipeline, 15)
    assert r14["overview_source"] == "kaggle" and r14["has_overview"] == 1
    assert r15["poster_path_source"] == "kaggle" and r15["has_poster"] == 1


def test_missing_or_broken_posters_are_flagged(pipeline):
    assert row(pipeline, 16)["has_poster"] == 0
    assert row(pipeline, 17)["has_poster"] == 0
    assert row(pipeline, 11)["has_poster"] == 1 and row(pipeline, 11)["poster_file"] == "posters/11.jpg"


def test_unknown_tmdb_and_missing_id_movies_are_kept_and_flagged(pipeline):
    assert len(pipeline.movies) == 26
    r404 = pipeline.movies[pipeline.movies["movie_id"] == 25].iloc[0]
    rnoid = pipeline.movies[pipeline.movies["movie_id"] == 26].iloc[0]
    for r in (r404, rnoid):
        assert r["in_tmdb"] == 0 and r["n_modalities"] == 0
    assert pipeline.stage("preprocess_catalog")["tmdb_id_missing_in_links"] == 1
    assert pipeline.stage("preprocess_meta")["tmdb_404"] == 1


def test_genres_are_unified_and_compared(pipeline):
    m = pipeline.movies
    r = row(pipeline, 12)
    assert list(r["genres_unified"]) == ["Action", "Science Fiction"]
    assert r["g_action"] == 1 and r["g_science_fiction"] == 1 and r["g_comedy"] == 0
    assert r["is_imax"] == 1
    assert pipeline.stage("preprocess_meta")["unmapped_tmdb_genres"] == {"Reality": 1}
    assert pipeline.stage("align")["genre_mean_jaccard"] == pytest.approx(1.0)
    assert set(genre_columns()) <= set(m.columns)


def test_language_runtime_and_year_normalization(pipeline):
    assert row(pipeline, 19)["original_language"] == "zh"
    assert row(pipeline, 20)["original_language"] == "und"
    assert row(pipeline, 18)["runtime_missing"] == 1 and pd.isna(row(pipeline, 18)["runtime_min"])
    assert row(pipeline, 11)["runtime_min"] == 100
    assert row(pipeline, 11)["release_year"] == 1991 and row(pipeline, 11)["title"] == "Movie 11"


def test_rating_splits_are_temporal_and_cold_items_isolated(pipeline):
    r = pipeline.ratings
    assert not r.duplicated(["user_id", "movie_id"]).any()
    tr, va, te = (r[r["split"] == s]["timestamp"] for s in ("train", "val", "test"))
    assert tr.max() <= va.min() and va.max() <= te.min()
    cold = r[r["split"] == "cold_item_test"]
    cold_movies = set(cold["movie_id"])
    assert len(cold_movies) == pipeline.stage("split")["cold_items"] >= 1
    assert not set(r[r["split"] != "cold_item_test"]["movie_id"]) & cold_movies  # aucune fuite
    assert not cold["item_in_train"].any()
    assert set(r["split"]) == {"train", "val", "test", "cold_item_test"}


def test_pipeline_succeeds_and_writes_reports(pipeline):
    assert pipeline.exit_code == 0
    rep = read_json(pipeline.reports / "quality_report.json")
    assert rep["volumes"]["movies"] == 26
    assert rep["validation"]["processed"]["n_errors"] == 0
    md = (pipeline.reports / "quality_report.md").read_text(encoding="utf-8")
    assert "not endorsed or certified by TMDB" in md
    assert (pipeline.reports / "quality_report.html").exists() and (pipeline.reports / "data_card.md").exists()
    man = json.loads((pipeline.reports.parent / "processed" / "manifest.json").read_text(encoding="utf-8"))
    assert man["seed"] == 42 and "movies_columns" in man
