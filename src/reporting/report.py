"""Étape `report` : rapport de qualité (JSON, Markdown, HTML autonome), profil, dérive, data card.

Sorties (paths.reports) :
  profile.json            profil versionné (métrique DVC) — base de la détection de dérive
  drift.json              comparaison avec drift.reference (si elle existe)
  quality_report.json/.md synthèse lisible par machine / par humain
  quality_report.html     rapport HTML autonome : tableau de bord des étapes, contrôles, distributions, dérive
  data_card.md            fiche de données : sources, licences, volumes, biais connus, prétraitements
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import read_json, write_json, write_text
from ..common.stage import QualityGateError, StageContext, git_commit, main_for
from .datacard import render_data_card
from .drift import build_profile, compare

STAGE_ORDER = ["ingest_movielens", "ingest_kaggle", "ingest_tmdb", "ingest_posters", "ingest_external", "validate_raw",
               "preprocess_catalog", "preprocess_text", "preprocess_images", "preprocess_meta", "preprocess_ratings",
               "align", "split", "validate_processed", "extract_features", "report"]
KEY_METRICS = {
    "ingest_movielens": ["movies", "ratings", "users"], "ingest_tmdb": ["found_200", "not_found_404"],
    "ingest_posters": ["manifest_records"],
    "ingest_external": ["movies", "ratings", "users", "new_users", "posters"], "ingest_kaggle": ["available", "rows_clean"],
    "validate_raw": ["n_checks", "n_errors", "n_warnings"], "preprocess_catalog": ["movies", "tmdb_id_missing_in_links"],
    "preprocess_text": ["rows", "has_overview_rate", "rejected_too_short"],
    "preprocess_images": ["rows", "has_poster_rate", "decode_failed", "manifest_records_outside_catalog"],
    "preprocess_meta": ["rows", "found_in_tmdb", "tmdb_404"],
    "preprocess_ratings": ["ratings_kept", "rejection_rate"], "align": ["movies", "join_rate_text", "join_rate_image"],
    "split": ["mode", "cold_items", "split_sizes"], "validate_processed": ["n_checks", "n_errors", "n_warnings"],
    "extract_features": ["backend", "dims"],
}
NOTES = [
    "popularity, vote_average et vote_count sont des instantanés TMDB à la date tmdb_fetched_at : ils reflètent "
    "l'attention actuelle, pas celle de la période des notes (fuite temporelle possible). Exclus des "
    "représentations méta (src/features/extract.py).",
    "Le dataset Kaggle est un instantané TMDB de 2017 dont les notes viennent de MovieLens : ce n'est pas une "
    "source indépendante.",
    "This product uses the TMDB API but is not endorsed or certified by TMDB.",
]


def _stages(reports: Path) -> list[dict[str, Any]]:
    out = []
    for name in STAGE_ORDER:
        f = reports / "stages" / f"{name}.json"
        out.append(read_json(f) if f.exists() else {"stage": name, "status": "not_run", "metrics": {}})
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{100 * v:.1f} %" if 0 <= v <= 1 else f"{v:,.2f}".replace(",", " ")
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}".replace(",", " ")
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    return str(v)


def _bars(edges: list[float], p: list[float], ref: list[float] | None = None) -> str:
    """Histogramme SVG inline (barres = version courante, trait = référence)."""
    w, h, n = 260, 70, len(p)
    bw = w / n
    top = max(max(p), max(ref) if ref else 0) or 1
    parts = [f'<svg viewBox="0 0 {w} {h + 14}" class="hist" role="img">']
    for i, v in enumerate(p):
        bh = h * v / top
        parts.append(f'<rect x="{i * bw + 1:.1f}" y="{h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" class="bar">'
                     f'<title>[{edges[i]:g} ; {edges[i + 1]:g}[ : {100 * v:.1f} %</title></rect>')
    if ref:
        pts = " ".join(f"{(i + 0.5) * bw:.1f},{h - h * v / top:.1f}" for i, v in enumerate(ref))
        parts.append(f'<polyline points="{pts}" class="ref"/>')
    parts.append(f'<text x="0" y="{h + 12}" class="axis">{edges[0]:g}</text>'
                 f'<text x="{w}" y="{h + 12}" class="axis" text-anchor="end">≥ {edges[-2]:g}</text></svg>')
    return "".join(parts)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--ok:#1a7f4b;--warn:#a86400;
--err:#b42318;--accent:#2f5bd3;--bar:#7b96e0}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--card:#1c1c1f;--ink:#ececee;--muted:#9a9aa2;--line:#2e2e33;
--ok:#4cc38a;--warn:#f0a64a;--err:#ff6b5e;--accent:#8aa8ff;--bar:#5b74c4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif}main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 10px}.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile b{display:block;font-size:20px}.tile span{color:var(--muted);font-size:12px}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;
color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.scroll{overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid currentColor}
.ok{color:var(--ok)}.warn{color:var(--warn)}.error,.failed,.failed_quality_gate,.alert{color:var(--err)}.not_run{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}.hist{width:100%;height:auto}
.bar{fill:var(--bar)}.ref{fill:none;stroke:var(--ink);stroke-width:1.5;stroke-dasharray:3 2}.axis{font-size:9px;fill:var(--muted)}
details summary{cursor:pointer;color:var(--accent)}code{font-size:12px}
"""


def render_html(ctx_info: dict[str, Any], stages: list[dict], suites: dict[str, dict], profile: dict,
                drift: dict | None, ref: dict | None) -> str:
    e = html.escape
    overall = "failed" if any(s["summary"]["status"] == "failed" for s in suites.values()) else "ok"
    v, c = profile["volumes"], profile["coverage"]
    tiles = [("Films", v["movies"]), ("Notes", v["ratings"]), ("Utilisateurs", v["users"]),
             ("Densité", f"{100 * v['density']:.2f} %"), ("Synopsis", c["has_overview"]), ("Affiche", c["has_poster"]),
             ("Méta", c["has_meta"]), ("3 modalités", c["all_modalities"])]
    out = [f"<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>Rapport qualité</title><style>{CSS}</style></head>"
           f"<body><main><h1>Rapport de qualité des données <span class='badge {overall}'>"
           f"{'contrôles OK' if overall == 'ok' else 'échec'}</span></h1>"
           f"<div class='muted'>Généré le {e(ctx_info['generated_at'])} · mode <code>{e(ctx_info['mode'])}</code> · "
           f"commit <code>{e(str(ctx_info['commit']))}</code> · MovieLens <code>{e(ctx_info['movielens'])}</code></div>",
           "<h2>Volumes et couverture des modalités</h2><div class='tiles'>"]
    out += [f"<div class='card tile'><b>{e(_fmt(val))}</b><span>{e(k)}</span></div>" for k, val in tiles]
    src = profile.get("sources", {})
    if src:
        out.append("</div><h2>Sources des données</h2><div class='card scroll'><table><tr><th>Source</th>"
                   "<th class='num'>Films</th><th class='num'>Notes</th><th>Notes par segment</th></tr>")
        for name in sorted(set(src.get("movies", {})) | set(src.get("ratings", {}))):
            segs = src.get("ratings_by_split", {}).get(name, {})
            out.append(f"<tr><td><code>{e(name)}</code></td><td class='num'>{_fmt(src['movies'].get(name, 0))}</td>"
                       f"<td class='num'>{_fmt(src['ratings'].get(name, 0))}</td><td>{e(_fmt(segs))}</td></tr>")
        out.append("</table><div class='muted'>Segment <code>new</code> : notes arrivées après le gel du découpage "
                   "(configs/split_spec.json), ni apprises ni évaluées tant que la spécification n'est pas re-gelée."
                   "</div>")
    out.append("</div><h2>Tableau de bord du pipeline</h2><div class='card scroll'><table><tr><th>Étape</th>"
               "<th>Statut</th><th class='num'>Durée</th><th>Heure (UTC)</th><th>Indicateurs</th></tr>")
    for s in stages:
        keys = KEY_METRICS.get(s["stage"], [])
        mets = "; ".join(f"{k} = {_fmt(s['metrics'][k])}" for k in keys if k in s.get("metrics", {}))
        dur = f"{s['duration_s']:.1f} s" if "duration_s" in s else "—"
        out.append(f"<tr><td><code>{e(s['stage'])}</code></td><td><span class='badge {e(s['status'])}'>"
                   f"{e(s['status'])}</span></td><td class='num'>{dur}</td><td class='muted'>{e(s.get('started_at', '—')[11:19])}"
                   f"</td><td>{e(mets)}</td></tr>")
    out.append("</table></div>")

    for name, suite in suites.items():
        sm = suite["summary"]
        cats = " · ".join(f"{k} {x['passed']}/{x['passed'] + x['failed']}" for k, x in sm["by_category"].items())
        out.append(f"<h2>Contrôles « {e(name)} » <span class='badge {sm['status'].replace('passed', 'ok')}'>"
                   f"{sm['n_passed']}/{sm['n_checks']}</span></h2><div class='card'><div class='muted'>{e(cats)}</div>")
        bad = [r for r in suite["checks"] if not r["passed"]]
        if bad:
            out.append("<table><tr><th>Contrôle</th><th>Sévérité</th><th>Observé</th><th>Seuil</th></tr>")
            out += [f"<tr><td>{e(r['name'])}<div class='muted'>{e(r['detail'])}</div></td><td class='{r['severity']}'>"
                    f"{r['severity']}</td><td><code>{e(json.dumps(r['observed'], ensure_ascii=False))}</code></td>"
                    f"<td><code>{e(json.dumps(r['threshold'], ensure_ascii=False))}</code></td></tr>" for r in bad]
            out.append("</table>")
        out.append("<details><summary>Tous les contrôles</summary><div class='scroll'><table><tr><th>Catégorie</th>"
                   "<th>Contrôle</th><th>Résultat</th><th>Observé</th></tr>")
        out += [f"<tr><td class='muted'>{r['category']}</td><td>{e(r['name'])}</td><td class='"
                f"{'ok' if r['passed'] else r['severity']}'>{'ok' if r['passed'] else 'échec'}</td><td><code>"
                f"{e(json.dumps(r['observed'], ensure_ascii=False))[:160]}</code></td></tr>" for r in suite["checks"]]
        out.append("</table></div></details></div>")

    out.append("<h2>Distributions</h2><div class='muted'>Barres : version courante · pointillés : référence "
               "(si présente)</div><div class='grid'>")
    for name, h in profile["histograms"].items():
        r = (ref or {}).get("histograms", {}).get(name)
        out.append(f"<div class='card'><b>{e(name)}</b>{_bars(h['edges'], h['p'], r['p'] if r and r['edges'] == h['edges'] else None)}</div>")
    out.append("</div>")

    out.append("<h2>Dérive par rapport à la référence</h2><div class='card'>")
    if drift is None:
        out.append("<span class='muted'>Aucune référence. Pour en créer une : "
                   "<code>python -m src.reporting.drift --promote</code>.</span>")
    else:
        out.append(f"<p>Statut : <span class='badge {drift['status']}'>{drift['status']}</span></p><table><tr>"
                   "<th>Variable</th><th>Indicateur</th><th class='num'>Valeur</th><th>Niveau</th></tr>")
        out += [f"<tr><td>{e(r['variable'])}</td><td>{r['kind']}</td><td class='num'>{r['value']}</td>"
                f"<td class='{r['level']}'>{r['level']}</td></tr>" for r in drift["rows"]]
        out.append("</table>")
    out.append("</div><h2>Avertissements</h2><div class='card'><ul>")
    out += [f"<li>{e(n)}</li>" for n in NOTES]
    out.append("</ul></div></main></body></html>")
    return "".join(out)


def render_md(info: dict, suites: dict, profile: dict, drift: dict | None) -> str:
    v, c = profile["volumes"], profile["coverage"]
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    lines = ["# Rapport qualité des données", "", f"Généré le {info['generated_at']} · mode `{info['mode']}` · "
             f"MovieLens `{info['movielens']}`", "", "## Volumes", "",
             f"- Films : {v['movies']} · notes : {v['ratings']} · utilisateurs : {v['users']} · densité : {pct(v['density'])}",
             "", "## Couverture des modalités", "",
             f"- Synopsis : {pct(c['has_overview'])} · affiche : {pct(c['has_poster'])} · méta : {pct(c['has_meta'])}"
             f" · trois modalités : {pct(c['all_modalities'])}", "", "## Contrôles", ""]
    for name, s in suites.items():
        sm = s["summary"]
        lines.append(f"- `{name}` : {sm['n_passed']}/{sm['n_checks']} réussis, {sm['n_errors']} erreur(s), "
                     f"{sm['n_warnings']} avertissement(s)")
    lines += ["", "## Dérive", "", f"- {drift['status'] if drift else 'pas de référence'}", "", "## Avertissements", ""]
    lines += [f"- {n}" for n in NOTES]
    return "\n".join(lines) + "\n"


def stage(ctx: StageContext) -> None:
    p, rep = ctx.params, ctx.reports
    movies = pd.read_parquet(ctx.processed / "movies.parquet")
    ratings = pd.read_parquet(ctx.processed / "ratings.parquet")
    profile = build_profile(movies, ratings)
    write_json(rep / "profile.json", profile)

    ref_path = Path(p["drift"]["reference"])
    ref = read_json(ref_path) if ref_path.exists() else None
    drift = compare(ref, profile, p["drift"]) if ref else None
    write_json(rep / "drift.json", drift or {"status": "no_reference", "rows": []})

    suites = {n: read_json(rep / "validation" / f"{n}.json") for n in ("raw", "external", "processed")
              if (rep / "validation" / f"{n}.json").exists()}
    info = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "mode": p["ingestion"]["mode"],
            "commit": git_commit(), "movielens": p["movielens"]["version"]}
    stages = _stages(rep)
    write_json(rep / "quality_report.json", {**info, "volumes": profile["volumes"], "coverage": profile["coverage"],
                                              "summary": profile["summary"],
                                              "validation": {n: s["summary"] for n, s in suites.items()},
                                              "drift_status": drift["status"] if drift else "no_reference",
                                              "stages": {s["stage"]: s["status"] for s in stages}, "notes": NOTES})
    write_text(rep / "quality_report.md", render_md(info, suites, profile, drift))
    write_text(rep / "quality_report.html", render_html(info, stages, suites, profile, drift, ref))
    write_text(rep / "data_card.md", render_data_card(p, profile, stages, suites))
    ctx.metric("drift_status", drift["status"] if drift else "no_reference")
    ctx.metric("html", (rep / "quality_report.html").as_posix())
    if drift and drift["status"] == "alert" and p["drift"]["fail_on_alert"]:
        raise QualityGateError("dérive forte détectée par rapport à la référence (drift.fail_on_alert=true)")


if __name__ == "__main__":
    main_for("report", stage, "Rapport de qualité, profil, dérive et data card")
