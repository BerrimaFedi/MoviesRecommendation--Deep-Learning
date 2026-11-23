"""Fiche de données (data card) régénérée à chaque exécution : les chiffres viennent du profil."""
from __future__ import annotations

from typing import Any


def render_data_card(p: dict[str, Any], profile: dict[str, Any], stages: list[dict], suites: dict[str, dict]) -> str:
    v, c, s = profile["volumes"], profile["coverage"], profile["summary"]
    split = profile["categorical"]["split"]
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    lang = profile["categorical"]["lang_cat"]
    top_lang = ", ".join(f"{k} {pct(x)}" for k, x in sorted(lang.items(), key=lambda kv: -kv[1])[:4])
    genres = profile["categorical"]["genres"]
    top_genres = ", ".join(f"{k} {pct(x)}" for k, x in sorted(genres.items(), key=lambda kv: -kv[1])[:5])
    val = "; ".join(f"{n} : {x['summary']['n_passed']}/{x['summary']['n_checks']}" for n, x in suites.items())
    synthetic = p["ingestion"]["mode"] == "synthetic"
    srcs = profile.get("sources", {})
    n_ext_m, n_ext_r = srcs.get("movies", {}).get("external", 0), srcs.get("ratings", {}).get("external", 0)
    ext_line = (f"| Inbox externe (`{p['paths']['inbox']}/`) | {n_ext_m} films, {n_ext_r} notes déposés par un tiers | "
                "fournies par le déposant ; contrôlées par `ingest_external` | segment `new` si le découpage est figé |\n"
                if n_ext_m or n_ext_r else "")
    return f"""# Fiche de données — Projet 10, recommandation multimodale de films

> Fichier généré par l'étape `report` (`src/reporting/datacard.py`) — ne pas éditer à la main.
> Mode d'ingestion : **{p['ingestion']['mode']}**{' — JEU SYNTHÉTIQUE, chiffres sans valeur statistique' if synthetic else ''}.

## 1. Motivation et usage prévu

Recommander des films à des utilisateurs MovieLens, y compris des films **jamais notés** (démarrage à froid),
à partir de trois modalités de contenu : affiche (image), synopsis (texte), métadonnées (tabulaire).
Usage strictement académique. **Hors périmètre** : usage commercial, redistribution des affiches et synopsis.

## 2. Sources, licences et conditions

| Source | Contenu | Licence / conditions | Version |
|---|---|---|---|
| MovieLens (GroupLens) | notes, catalogue, liens IMDb/TMDB | usage de recherche, redistribution interdite, citation requise (Harper & Konstan, 2015) | `{p['movielens']['version']}` |
| API TMDB | synopsis, affiches w342, durée, langue, genres, popularité | API Terms of Use : attribution obligatoire, cache ≤ 6 mois, usage commercial soumis à accord — l'entraînement de modèles ML peut être assimilé à un usage commercial, à valider avec l'encadrant | langue `{p['tmdb']['language']}`, taille `{p['tmdb']['poster_size']}` |
| Kaggle « The Movies Dataset » | secours : synopsis, durée, chemin d'affiche | licence du dépôt Kaggle (CC0 déclarée) ; instantané TMDB de juillet 2017, pas une source indépendante | `{p['kaggle']['dataset']}` |
{ext_line}
*This product uses the TMDB API but is not endorsed or certified by TMDB.*

## 3. Composition (version courante)

- **{v['movies']:,} films**, **{v['ratings']:,} notes**, **{v['users']:,} utilisateurs** — densité {pct(v['density'])}.
- Couverture : synopsis {pct(c['has_overview'])} · affiche {pct(c['has_poster'])} · métadonnées {pct(c['has_meta'])} ·
  les trois {pct(c['all_modalities'])}. Durée manquante {pct(c['runtime_missing'])}, année manquante {pct(c['year_missing'])}.
- Synopsis : médiane de {s['overview_words_median']:.0f} mots. Affiches : 342 px de large, ratio ≈ 1,5.
- Notes : moyenne {s['rating_mean']:.2f} sur une échelle 0,5–5 au pas de 0,5. Médiane de
  {s['ratings_per_item_median']:.0f} note(s) par film ; {pct(s['items_with_at_most_5_ratings'])} des films ont ≤ 5 notes.
- Langues : {top_lang}. Genres les plus fréquents : {top_genres}.
- Découpage : {', '.join(f'{k} {pct(x)}' for k, x in split.items())}.

## 4. Prétraitements (une étape DVC chacun)

1. `preprocess_catalog` — genres MovieLens → vocabulaire canonique de 20 genres (Children→Family, Sci-Fi→Science
   Fiction, IMAX → indicateur `is_imax`), année extraite du titre, identifiant IMDb normalisé.
2. `preprocess_text` — Unicode {p['preprocessing']['text']['unicode_form']}, caractères de contrôle supprimés, espaces
   normalisés ; synopsis < {p['preprocessing']['text']['min_len_chars']} caractères = absent ; secours Kaggle.
3. `preprocess_images` — seules les affiches valides (JPEG RGB décodable, côté ≥
   {p['preprocessing']['image']['min_side']} px) sont retenues ; empreinte SHA-1 conservée.
4. `preprocess_meta` — langue ISO 639-1 (cn→zh, xx→und), durée 0 = inconnue, popularité/votes TMDB seulement.
5. `preprocess_ratings` — doublons (utilisateur, film) supprimés, notes hors domaine rejetées.
6. `align` — jointure movie_id ↔ tmdb_id ; aucun film supprimé, absences marquées (`has_*`, `modality_mask`).
7. `split` — {pct(p['split']['cold_item_frac'])} des films en démarrage à froid strict, puis découpage chronologique
   {p['split']['train_frac']:.0%}/{p['split']['val_frac']:.0%}/{1 - p['split']['train_frac'] - p['split']['val_frac']:.0%},
   graine {p['project']['seed']}.

Contrôles de qualité de cette version : {val}.

## 5. Biais et limites connus

- **Population** : {v['users']:,} utilisateurs MovieLens volontaires, non représentatifs du grand public ;
  pour latest-small, notes de 1996 à 2018.
- **Popularité** : distribution à longue traîne ; les films très notés dominent l'apprentissage collaboratif.
- **Langue et culture** : catalogue majoritairement anglophone et américain ; synopsis tous en anglais (`en-US`).
- **Fuite temporelle** : popularité et votes TMDB sont des instantanés *actuels*, postérieurs aux notes : exclus des
  représentations du modèle.
- **Genres** : accord MovieLens/TMDB moyen (Jaccard {s['genre_mean_jaccard']:.2f}) ; l'union des deux est utilisée.
- **Affiches** : une affiche reflète le marketing d'une époque et d'un marché ; elle peut changer dans le temps sur TMDB.

## 6. Maintenance

Reconstruction : `dvc repro` (ou `dvc pull` pour récupérer les données versionnées). Cache TMDB à rafraîchir tous les
6 mois au plus (`ingest_tmdb`). Contact : binôme du projet (voir README).
"""
