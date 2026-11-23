"""Tests du prototype de fusion : dimensions, modalités manquantes, gradients, batch réel du pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.fusion.data import ItemStore, RatingBatches  # noqa: E402
from src.fusion.model import FusionConfig, MultimodalRecommender, info_nce  # noqa: E402
from src.fusion.modules import GatedMultimodalFusion  # noqa: E402

B, D = 8, 32


def _cfg(**kw) -> FusionConfig:
    return FusionConfig(d_text=12, d_image=20, d_meta=6, n_items=10, n_users=5, d_model=D, meta_hidden=16, **kw)


def _batch(mask=None, item_idx=None) -> dict:
    g = torch.Generator().manual_seed(0)
    return {"text": torch.randn(B, 12, generator=g), "image": torch.randn(B, 20, generator=g),
            "meta": torch.randn(B, 6, generator=g),
            "mask": torch.ones(B, 3) if mask is None else mask,
            "item_idx": torch.randint(1, 10, (B,), generator=g) if item_idx is None else item_idx,
            "user_idx": torch.randint(1, 5, (B,), generator=g), "rating": torch.rand(B, generator=g) * 4.5 + 0.5}


@pytest.mark.parametrize("fusion", ["gmu", "concat", "attention"])
def test_forward_shapes_and_backward(fusion):
    model = MultimodalRecommender(_cfg(fusion=fusion))
    b = _batch()
    out = model(b)
    assert out["score"].shape == (B,) and out["e"].shape == (B, D)
    assert out["unimodal"].shape == (B, 4) and out["z"].shape == (B, 4, D)
    loss = model.loss(out, b["rating"])["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.proj["text"].net[-2].weight.grad is not None


def test_gate_ignores_absent_modalities_and_sums_to_one():
    fus = GatedMultimodalFusion(4, D)
    p = torch.randn(B, 4, D)
    mask = torch.tensor([[1, 0, 1, 0]] * B, dtype=torch.float)
    _, z = fus(p, mask)
    assert torch.allclose(z.sum(dim=1), torch.ones(B, D), atol=1e-5)
    assert torch.all(z[:, 1] == 0) and torch.all(z[:, 3] == 0)


def test_all_modalities_absent_still_produces_finite_output():
    model = MultimodalRecommender(_cfg()).eval()
    out = model(_batch(mask=torch.zeros(B, 3), item_idx=torch.zeros(B, dtype=torch.long)))
    assert torch.isfinite(out["score"]).all()


def test_cold_items_never_use_the_id_embedding():
    model = MultimodalRecommender(_cfg()).eval()
    out = model(_batch(item_idx=torch.zeros(B, dtype=torch.long)))
    assert model.modality_contribution(out)["id"] == 0.0


def test_modality_dropout_only_in_training():
    model = MultimodalRecommender(_cfg(modality_dropout=0.9, id_dropout=0.9))
    b = _batch()
    model.train()
    assert model(b)["mask"].mean() < 1.0
    model.eval()
    assert model(b)["mask"].mean() == 1.0


def test_info_nce_prefers_aligned_pairs():
    a = torch.randn(16, D)
    assert info_nce(a, a, 0.07) < info_nce(a, torch.randn(16, D), 0.07)


def test_forward_on_real_batch_from_pipeline(sample_pipeline):
    """Batch issu des sorties réelles du pipeline (features/items.npz + ratings.parquet)."""
    p = sample_pipeline
    store = ItemStore.load(Path(p["paths"]["features"]))
    data = RatingBatches(store, pd.read_parquet(Path(p["paths"]["processed"]) / "ratings.parquet"))
    cfg = FusionConfig.from_params(p["fusion"], store.dims, data.n_items, data.n_users, global_mean=data.global_mean)
    model = MultimodalRecommender(cfg)
    b = data.batch("train", 64)
    out = model(b)
    assert out["e"].shape == (len(b["rating"]), p["fusion"]["d_model"])
    assert b["text"].shape[1] == store.dims["text"] and b["image"].shape[1] == store.dims["image"]
    cold = data.batch("cold_item_test", 64)
    assert (cold["item_idx"] == 0).all()                      # aucun film froid n'a d'identifiant appris
    model.loss(out, b["rating"])["loss"].backward()
