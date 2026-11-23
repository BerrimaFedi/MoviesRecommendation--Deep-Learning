"""Étape `extract_features` : représentations par modalité, mises en cache pour la fusion.

Sortie : paths.features/items.npz — movie_id (N), text (N×d_t), image (N×d_i), meta (N×d_m),
mask (N×3 : texte, image, méta présents) — et paths.features/features.json (dimensions, noms, backend).

Deux backends (features.backend) :
  light       déterministe, sans téléchargement : sac de mots haché signé (texte), vignette RGB
              (image). Sert à la CI et à vérifier la chaîne de bout en bout ; pas la qualité finale.
  pretrained  encodeurs figés de la Tâche 4 : Sentence-BERT all-MiniLM-L6-v2 (384) et ResNet-50
              ImageNet (2048, couche de pooling). Dépendances : requirements-ml.txt.

Méta-données : multi-hot des genres, langue (one-hot lang_cat), année et durée standardisées,
indicateurs de valeurs manquantes. La moyenne / l'écart-type sont calculés sur les seuls films
vus à l'entraînement (pas de fuite depuis val/test/cold). Popularité et votes sont exclus : instantanés
TMDB postérieurs aux notes (fuite temporelle documentée dans la data card).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..common.io import write_json
from ..common.stage import StageContext, main_for
from ..preprocessing.normalize import genre_columns

_TOKEN = re.compile(r"[a-zà-ÿ0-9']+")


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def hashed_bow(texts: list[str | None], dim: int) -> np.ndarray:
    """Sac de mots haché signé (astuce du hachage), pondération log(1+tf), normalisé L2."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            continue
        for tok in _TOKEN.findall(t.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            out[i, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    return _l2(np.sign(out) * np.log1p(np.abs(out)))


def thumbnails(files: list[Path | None], size: int) -> np.ndarray:
    out = np.zeros((len(files), size * size * 3), dtype=np.float32)
    for i, f in enumerate(files):
        if f is None:
            continue
        with Image.open(f) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel() / 255
    present = np.any(out != 0, axis=1)
    out[present] -= out[present].mean(axis=1, keepdims=True)
    return _l2(out)


def pretrained_text(texts: list[str | None], model_name: str, batch: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # requirements-ml.txt

    model = SentenceTransformer(model_name)
    emb = model.encode([t or "" for t in texts], batch_size=batch, normalize_embeddings=True, show_progress_bar=True)
    emb[[not t for t in texts]] = 0.0
    return emb.astype(np.float32)


def pretrained_image(files: list[Path | None], batch: int) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50  # requirements-ml.txt

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()  # redimension 232, recadrage 224, normalisation ImageNet
    out = np.zeros((len(files), 2048), dtype=np.float32)
    idx = [i for i, f in enumerate(files) if f is not None]
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            x = torch.stack([tf(Image.open(files[i]).convert("RGB")) for i in chunk])
            out[chunk] = model(x).numpy()
    return out


def meta_matrix(movies: pd.DataFrame, train_ids: set[int]) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    g = genre_columns()
    cols.append(movies[g].to_numpy(np.float32))
    names += g
    langs = sorted(movies["lang_cat"].unique())
    cols.append(np.stack([(movies["lang_cat"] == lang).to_numpy(np.float32) for lang in langs], axis=1))
    names += [f"lang_{lang}" for lang in langs]
    fit = movies["movie_id"].isin(train_ids)
    for col, transform in (("release_year", lambda s: s.astype("float")),
                           ("runtime_min", lambda s: np.log1p(s.astype("float")))):
        v = transform(movies[col])
        mu, sd = v[fit].mean(), v[fit].std() or 1.0
        cols.append(((v - mu) / sd).fillna(0).to_numpy(np.float32)[:, None])
        cols.append(v.isna().to_numpy(np.float32)[:, None])
        names += [f"{col}_z", f"{col}_missing"]
    cols.append(movies[["is_imax"]].to_numpy(np.float32))
    names.append("is_imax")
    return np.concatenate(cols, axis=1), names


def poster_path(poster_file: str, posters_root: Path, interim: Path) -> Path:
    """« posters/<id>.jpg » (TMDB, sous data/raw) ou « external/posters/<id>.jpg » (inbox, sous interim)."""
    return (interim if poster_file.startswith("external/") else posters_root) / poster_file


def extract(params: dict[str, Any], movies: pd.DataFrame, ratings: pd.DataFrame, posters_root: Path,
            interim: Path | None = None) -> dict:
    fp = params["features"]
    interim = interim or Path(params["paths"]["interim"])
    texts = [t if h else None for t, h in zip(movies["overview"], movies["has_overview"])]
    files = [poster_path(f, posters_root, interim) if h else None
             for f, h in zip(movies["poster_file"], movies["has_poster"])]
    if fp["backend"] == "light":
        text, image = hashed_bow(texts, fp["text_dim"]), thumbnails(files, fp["image_thumb"])
    elif fp["backend"] == "pretrained":
        text = pretrained_text(texts, fp["pretrained"]["text_model"], fp["batch_size"])
        image = pretrained_image(files, fp["batch_size"])
    else:
        raise ValueError(f"backend inconnu : {fp['backend']}")
    train_ids = set(ratings.loc[ratings["split"] == "train", "movie_id"])
    meta, meta_names = meta_matrix(movies, train_ids)
    mask = movies[["has_overview", "has_poster", "has_meta"]].to_numpy(np.int8)
    return {"movie_id": movies["movie_id"].to_numpy(np.int64), "text": text, "image": image, "meta": meta,
            "mask": mask, "meta_names": meta_names}


def stage(ctx: StageContext) -> None:
    movies = pd.read_parquet(ctx.processed / "movies.parquet")
    ratings = pd.read_parquet(ctx.processed / "ratings.parquet")
    posters_root = ctx.paths["raw_posters"].parent  # poster_file = « posters/<id>.jpg »
    f = extract(ctx.params, movies, ratings, posters_root)
    out = ctx.paths["features"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "items.npz", **{k: v for k, v in f.items() if k != "meta_names"})
    info = {"backend": ctx.params["features"]["backend"], "n_items": len(f["movie_id"]),
            "dims": {k: int(f[k].shape[1]) for k in ("text", "image", "meta")}, "meta_names": f["meta_names"],
            "presence_rate": dict(zip(["text", "image", "meta"], f["mask"].mean(axis=0).round(4).tolist()))}
    write_json(out / "features.json", info)
    for k in ("backend", "n_items", "dims", "presence_rate"):
        ctx.metric(k, info[k])


if __name__ == "__main__":
    main_for("extract_features", stage, "Extrait et met en cache les représentations par modalité")
