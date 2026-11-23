"""Mini-cadre de contrôles de qualité : résultat typé, sévérité, porte de qualité.

Choix : un module maison plutôt que Great Expectations / Pandera. Le volume est faible (≈10⁴ films,
10⁵ notes), les contrôles inter-modalités et anti-fuite sont spécifiques au projet et s'écrivent plus
clairement en pandas, et l'on évite une dépendance lourde dont les versions suivent mal pandas 3.
Chaque contrôle produit un enregistrement JSON (observé, seuil, sévérité) exploité par le rapport HTML.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import to_jsonable, write_json
from ..common.stage import QualityGateError, StageContext

CATEGORIES = ("schema", "missing", "duplicates", "outliers", "inter_modal", "temporal", "leakage", "source")


@dataclass
class CheckResult:
    name: str
    category: str
    severity: str  # error | warn
    passed: bool
    observed: Any = None
    threshold: Any = None
    detail: str = ""


@dataclass
class Suite:
    name: str
    results: list[CheckResult] = field(default_factory=list)

    def check(self, name: str, passed: bool, *, category: str, severity: str = "error", observed: Any = None,
              threshold: Any = None, detail: str = "") -> bool:
        assert category in CATEGORIES, category
        self.results.append(CheckResult(name, category, severity, bool(passed), to_jsonable(observed),
                                        to_jsonable(threshold), detail))
        return bool(passed)

    def rate_at_most(self, name: str, bad: pd.Series | int, total: int | None = None, *, max_rate: float,
                     category: str, severity: str = "error", detail: str = "") -> bool:
        """Contrôle « taux de lignes en défaut ≤ seuil » (bad = masque booléen ou effectif)."""
        if isinstance(bad, pd.Series):
            total = len(bad) if total is None else total
            bad = int(bad.fillna(False).astype(bool).sum())
        rate = (bad / total) if total else 0.0
        return self.check(name, rate <= max_rate, category=category, severity=severity,
                          observed={"rate": round(rate, 6), "count": int(bad), "total": int(total or 0)},
                          threshold=max_rate, detail=detail)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warn"]

    def summary(self) -> dict[str, Any]:
        by_cat: dict[str, dict[str, int]] = {}
        for r in self.results:
            c = by_cat.setdefault(r.category, {"passed": 0, "failed": 0})
            c["passed" if r.passed else "failed"] += 1
        return {"suite": self.name, "n_checks": len(self.results), "n_passed": sum(r.passed for r in self.results),
                "n_errors": len(self.errors), "n_warnings": len(self.warnings), "by_category": by_cat,
                "status": "failed" if self.errors else "passed"}

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "checks": [asdict(r) for r in self.results]}

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    def enforce(self, ctx: StageContext) -> None:
        """Publie les statistiques, journalise les échecs et arrête le pipeline si une erreur est présente."""
        s = self.summary()
        for k in ("n_checks", "n_passed", "n_errors", "n_warnings"):
            ctx.metric(k, s[k])
        for r in self.warnings:
            ctx.warn(f"avertissement qualité : {r.name}", observed=r.observed, threshold=r.threshold)
        for r in self.errors:
            ctx.log.error(f"contrôle échoué : {r.name} (observé={r.observed}, seuil={r.threshold}) {r.detail}")
        if self.errors:
            names = ", ".join(r.name for r in self.errors)
            raise QualityGateError(f"{len(self.errors)} contrôle(s) critique(s) en échec dans « {self.name} » : {names}")
        ctx.info(f"{s['n_passed']}/{s['n_checks']} contrôles réussis, {s['n_warnings']} avertissement(s)")
