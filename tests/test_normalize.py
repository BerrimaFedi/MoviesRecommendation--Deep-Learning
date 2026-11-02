import asyncio
import time

from collecte.normalize import (clean_text, imdb_tt, jaccard, normalize_language, parse_ml_genres,
                                parse_tmdb_genres, split_title_year)
from collecte.tmdb import RateLimiter, load_cache


def test_movielens_genres_are_mapped_to_canonical_vocabulary():
    genres, imax = parse_ml_genres("Adventure|Children|Sci-Fi|Musical|IMAX")
    assert genres == ["Adventure", "Family", "Science Fiction", "Music"]
    assert imax is True


def test_movielens_no_genres_and_unknown_genre_are_reported():
    unmapped: dict[str, int] = {}
    assert parse_ml_genres("(no genres listed)", unmapped) == ([], False)
    assert parse_ml_genres("Action|Telenovela", unmapped)[0] == ["Action"]
    assert unmapped == {"Telenovela": 1}


def test_kaggle_stringified_genres_are_parsed():
    s = "[{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}, {'id': 1, 'name': 'Foreign'}]"
    assert parse_tmdb_genres(s) == ["Action", "Science Fiction"]
    assert parse_tmdb_genres("not a list") == []
    assert parse_tmdb_genres(None) == []


def test_tmdb_api_genre_dicts_are_parsed():
    assert parse_tmdb_genres([{"id": 10751, "name": "Family"}, {"id": 10402, "name": "Music"}]) == ["Family", "Music"]


def test_jaccard():
    assert jaccard(["Action"], ["Action"]) == 1.0
    assert jaccard(["Action", "Drama"], ["Action"]) == 0.5
    assert jaccard([], ["Action"]) is None


def test_language_normalization():
    assert normalize_language("EN") == "en"
    assert normalize_language("cn") == "zh"
    assert normalize_language("xx") == "und"
    assert normalize_language(None) == "und"
    assert normalize_language("pt-BR") == "pt"


def test_title_year_split():
    assert split_title_year("Toy Story (1995)") == ("Toy Story", 1995)
    assert split_title_year("City of Lost Children, The (Cité des enfants perdus, La) (1995)")[1] == 1995
    assert split_title_year("No year here") == ("No year here", None)


def test_imdb_id_is_zero_padded():
    assert imdb_tt(114709) == "tt0114709"
    assert imdb_tt(None) is None


def test_clean_text():
    assert clean_text("  a \n b  ") == "a b"
    assert clean_text("   ") is None
    assert clean_text(None) is None


def test_rate_limiter_spaces_requests():
    async def go():
        lim = RateLimiter(50.0)
        t0 = time.monotonic()
        await asyncio.gather(*(lim.wait() for _ in range(10)))
        return time.monotonic() - t0

    assert asyncio.run(go()) >= 0.16  # 9 intervalles de 20 ms


def test_cache_ignores_truncated_last_line(tmp_path):
    p = tmp_path / "movies.jsonl"
    p.write_text('{"tmdb_id": 1, "status": 200, "fetched_at": "2026-01-01T00:00:00+00:00", "data": {}}\n'
                 '{"tmdb_id": 2, "sta', encoding="utf-8")
    assert list(load_cache(p)) == [1]
