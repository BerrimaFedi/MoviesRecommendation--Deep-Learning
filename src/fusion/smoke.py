"""Vérification du prototype sur un VRAI batch issu du pipeline : dimensions, pertes, rétropropagation.

  python -m src.fusion.smoke                       # données réelles (configs/params.yaml)
  python -m src.fusion.smoke --params configs/params.ci.yaml --fusion concat
Le résultat est journalisé via src/training/tracking.py (MLflow si installé, sinon JSON local).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ..common.config import load_params
from ..training.tracking import Tracker
from .data import ItemStore, RatingBatches
from .model import FusionConfig, MultimodalRecommender


def run(params: dict, fusion: str = "gmu", batch_size: int = 256) -> dict:
    torch.manual_seed(params["project"]["seed"])
    paths = params["paths"]
    store = ItemStore.load(Path(paths["features"]))
    data = RatingBatches(store, pd.read_parquet(Path(paths["processed"]) / "ratings.parquet"))
    cfg = FusionConfig.from_params(params["fusion"], store.dims, data.n_items, data.n_users, fusion=fusion,
                                   global_mean=data.global_mean)
    model = MultimodalRecommender(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    report: dict = {"fusion": fusion, "n_params": sum(p.numel() for p in model.parameters()),
                    "n_users": data.n_users, "n_items": data.n_items, "dims": store.dims}

    model.train()
    b = data.batch("train", batch_size, seed=0)
    out = model(b)
    losses = model.loss(out, b["rating"])
    losses["loss"].backward()
    opt.step()
    report["shapes"] = {k: list(out[k].shape) for k in ("score", "unimodal", "e", "z", "p", "mask")}
    report["train_losses"] = {k: round(float(v.detach()), 4) for k, v in losses.items()}

    model.eval()
    with torch.no_grad():
        cold = data.batch("cold_item_test", batch_size, seed=1)
        oc = model(cold)
        report["cold_item_idx_all_zero"] = bool((cold["item_idx"] == 0).all())
        report["cold_rmse_untrained"] = round(float(((oc["score"] - cold["rating"]) ** 2).mean().sqrt()), 4)
        report["cold_modality_contribution"] = model.modality_contribution(oc)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--fusion", default="gmu", choices=["gmu", "concat", "attention"])
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    params = load_params(a.params)
    rep = run(params, a.fusion, a.batch_size)
    with Tracker(params, run_name=f"smoke-{a.fusion}") as t:
        t.log_params({"fusion": a.fusion, "batch_size": a.batch_size, **params["fusion"]})
        t.log_metrics({f"smoke_{k}": v for k, v in rep["train_losses"].items()})
        t.log_metrics({"smoke_cold_rmse_untrained": rep["cold_rmse_untrained"]})
        t.log_dict(rep, "smoke_report.json")
    for k, v in rep.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
