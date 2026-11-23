"""Suivi d'expériences : façade MLflow, avec repli sur des fichiers JSON si MLflow est absent.

Préparé dès la Tâche 3 pour que l'entraînement (Tâche 4) n'ait qu'à appeler :
    with Tracker(params, run_name="gmu-v1") as t:
        t.log_params(...); t.log_metrics({...}, step=epoch); t.log_dict(rapport, "eval.json")
Chaque run enregistre aussi le commit Git et l'empreinte de la configuration (traçabilité données ↔ modèle).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..common.config import params_hash
from ..common.io import to_jsonable
from ..common.stage import git_commit


class Tracker:
    def __init__(self, params: dict[str, Any], run_name: str) -> None:
        tp = params["tracking"]
        self.params, self.run_name = params, run_name
        self._mlflow = None
        if tp["backend"] == "mlflow":
            try:
                import mlflow  # requirements-ml.txt

                mlflow.set_tracking_uri(tp["tracking_uri"])
                mlflow.set_experiment(tp["experiment"])
                self._mlflow = mlflow
            except ImportError:
                pass
        self._dir = Path(tp["tracking_uri"]) / "local_runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_name}"
        self._record: dict[str, Any] = {"run_name": run_name, "params": {}, "metrics": {}, "artifacts": {}}

    def __enter__(self) -> "Tracker":
        tags = {"git_commit": git_commit() or "unknown", "params_hash": params_hash(self.params),
                "ingestion_mode": self.params["ingestion"]["mode"]}
        if self._mlflow:
            self._mlflow.start_run(run_name=self.run_name, tags=tags)
        self._record["tags"] = tags
        return self

    def log_params(self, d: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.log_params({k: str(v) for k, v in d.items()})
        self._record["params"].update(to_jsonable(d))

    def log_metrics(self, d: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)
        for k, v in d.items():
            self._record["metrics"].setdefault(k, []).append({"step": step, "value": float(v)})

    def log_dict(self, d: dict[str, Any], name: str) -> None:
        if self._mlflow:
            self._mlflow.log_dict(to_jsonable(d), name)
        self._record["artifacts"][name] = to_jsonable(d)

    def __exit__(self, *exc: Any) -> None:
        if self._mlflow:
            self._mlflow.end_run(status="FAILED" if exc[0] else "FINISHED")
        else:
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / "run.json").write_text(json.dumps(self._record, indent=2, ensure_ascii=False), encoding="utf-8")
