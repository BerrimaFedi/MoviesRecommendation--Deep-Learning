"""Assemble le rapport de la Tâche 3 (HTML) et l'imprime en PDF avec Edge/Chrome sans interface.

  python docs/report/build.py
Sorties : docs/Rapport_Tache3.html, docs/Rapport_Tache3.pdf
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

EXTRA_REFS = {
    9: "D. Kreuzberger, N. Kühl, S. Hirschl, « Machine Learning Operations (MLOps): Overview, Definition, and "
       "Architecture », <em>IEEE Access</em>, 11, 2023.",
    10: "Documentation officielle : DVC (dvc.org), Great Expectations (greatexpectations.io), GitHub Actions "
        "(docs.github.com/actions).",
    23: "F. M. Harper, J. A. Konstan, « The MovieLens Datasets: History and Context », <em>ACM Transactions on "
        "Interactive Intelligent Systems</em>, 5(4), 2015.",
}


def build() -> Path:
    main = (HERE / "main.html").read_text(encoding="utf-8")
    b1 = (HERE / "b1_etat_de_l_art.html").read_text(encoding="utf-8")
    cut = b1.index('<h3 id="refs-b1">')
    body, refs_html = b1[:cut], b1[cut:]
    refs = {int(n): txt.strip() for n, txt in re.findall(r'<li value="(\d+)">(.*?)</li>', refs_html, flags=re.S)}
    refs.update(EXTRA_REFS)
    ref_list = "\n".join(f'<li value="{n}">{refs[n]}</li>' for n in sorted(refs))
    refs_section = f'<h2 id="refs">7. Références</h2>\n<ol class="refs">\n{ref_list}\n</ol>'
    html = main.replace("<!--B1-->", '<div class="page-break"></div>\n' + body).replace("<!--REFS-->", refs_section)
    out = OUT / "Rapport_Tache3.html"
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html: Path) -> Path | None:
    candidates = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("chromium"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if exe is None:
        print("Aucun navigateur Chromium trouvé : ouvrez le HTML et imprimez-le en PDF.")
        return None
    pdf = html.with_suffix(".pdf")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], check=True, capture_output=True, timeout=120)
    return pdf


if __name__ == "__main__":
    h = build()
    print(h)
    print(to_pdf(h))
