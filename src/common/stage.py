"""Exécution uniforme d'une étape du pipeline : arguments, journaux structurés, statistiques.

Chaque étape est un module `python -m src.<paquet>.<module> [--params ...] [--set clé=valeur]`.
`run_stage` :
  - charge configs/params.yaml (+ surcharges) ;
  - journalise en console et en JSON lignes dans logs/pipeline.jsonl (une ligne par événement) ;
  - écrit reports/stages/<étape>.json : statut, durée, commit Git, empreinte des paramètres et les
    statistiques déclarées par l'étape (volumes, taux de rejet…) ; le tableau de bord les agrège ;
  - code de sortie 1 si un contrôle critique échoue (QualityGateError) : DVC arrête alors le pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_PARAMS, load_dotenv, load_params, params_hash
from .io import to_jsonable, write_json


class QualityGateError(RuntimeError):
    """Un contrôle de qualité critique a échoué : le pipeline doit s'arrêter."""


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "stage": getattr(record, "stage", record.name),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            payload["data"] = to_jsonable(extra)
        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Message lisible + données structurées abrégées."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        data = getattr(record, "data", None)
        if data:
            extra = json.dumps(to_jsonable(data), ensure_ascii=False)
            text += " " + (extra if len(extra) <= 400 else extra[:400] + "…")
        return text


def get_logger(stage: str, log_dir: Path | None = None) -> logging.LoggerAdapter:
    logger = logging.getLogger(f"pipeline.{stage}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ConsoleFormatter(f"[{stage}] %(levelname)s %(message)s"))
        logger.addHandler(console)
    # le fichier de journal suit la configuration de l'exécution courante (plusieurs pipelines par processus)
    target = (log_dir / "pipeline.jsonl").resolve() if log_dir is not None else None
    for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        if target is None or Path(h.baseFilename) != target:
            logger.removeHandler(h)
            h.close()
    if target is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)
    return logging.LoggerAdapter(logger, {"stage": stage}, merge_extra=True)  # Python ≥ 3.13


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class StageContext:
    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params
        self.paths = {k: Path(v) for k, v in params["paths"].items()}
        self.log = get_logger(name, self.paths["logs"])
        self.metrics: dict[str, Any] = {}

    def metric(self, key: str, value: Any) -> None:
        """Statistique publiée dans reports/stages/<étape>.json et dans le journal."""
        self.metrics[key] = to_jsonable(value)

    def info(self, msg: str, **data: Any) -> None:
        self.log.info(msg, extra={"data": data} if data else None)

    def warn(self, msg: str, **data: Any) -> None:
        self.log.warning(msg, extra={"data": data} if data else None)

    @property
    def interim(self) -> Path:
        return self.paths["interim"]

    @property
    def processed(self) -> Path:
        return self.paths["processed"]

    @property
    def reports(self) -> Path:
        return self.paths["reports"]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--params", default=str(DEFAULT_PARAMS), help="fichier de configuration YAML")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLÉ=VALEUR",
                    help="surcharge d'un paramètre, ex. --set tmdb.limit=200 (répétable)")
    return ap


def run_stage(name: str, fn: Callable[[StageContext], None], description: str = "",
              argv: list[str] | None = None, params: dict[str, Any] | None = None) -> int:
    """Point d'entrée commun. `params` permet aux tests d'appeler l'étape sans fichier."""
    load_dotenv()
    if params is None:
        args = build_arg_parser(description or name).parse_args(argv)
        params = load_params(args.params, args.overrides)
    ctx = StageContext(name, params)
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    status, error = "ok", None
    ctx.info("début de l'étape", params_hash=params_hash(params))
    try:
        fn(ctx)
    except QualityGateError as e:
        status, error = "failed_quality_gate", str(e)
        ctx.log.error(f"CONTRÔLE CRITIQUE ÉCHOUÉ : {e}")
    except Exception as e:  # noqa: BLE001 - on journalise toute erreur puis on la relance
        status, error = "error", f"{type(e).__name__}: {e}"
        ctx.log.exception("erreur inattendue")
        _write_stage_record(ctx, started, t0, status, error)
        raise
    _write_stage_record(ctx, started, t0, status, error)
    return 0 if status == "ok" else 1


def _write_stage_record(ctx: StageContext, started: datetime, t0: float, status: str, error: str | None) -> None:
    duration = round(time.perf_counter() - t0, 3)
    record = {
        "stage": ctx.name, "status": status, "error": error,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": duration, "git_commit": git_commit(), "params_hash": params_hash(ctx.params),
        "python": platform.python_version(), "ingestion_mode": ctx.params["ingestion"]["mode"],
        "metrics": ctx.metrics,
    }
    write_json(ctx.reports / "stages" / f"{ctx.name}.json", record)
    ctx.info("fin de l'étape", status=status, duration_s=duration, **ctx.metrics)


def main_for(name: str, fn: Callable[[StageContext], None], description: str = "") -> None:
    raise SystemExit(run_stage(name, fn, description))
