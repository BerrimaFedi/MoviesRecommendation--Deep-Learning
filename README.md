# Projet 10 — Recommandation multimodale de films

Binôme : Fedi Berrima & Nadia Akhsay · Module Deep Learning · 2026–2027

Recommander des films, y compris des films **jamais notés** (démarrage à froid), à partir de trois modalités :
**affiche** (image), **synopsis** (texte) et **métadonnées** (tabulaire), en s'appuyant sur MovieLens (notes) et TMDB
(contenu).

| Tâche | Contenu | Où |
|---|---|---|
| 1 | Sources, algorithmes, stratégie de collecte | `Rapport_Tache1.md`, `rapport_tache1.*` |
| 2 | Collecte et préparation des modalités | `src/ingestion`, `src/preprocessing` |
| **3** | **Pipeline DataOps automatisé + étude de la fusion** | `dvc.yaml`, `src/`, `tests/`, `docs/Rapport_Tache3.pdf` |

> Ce produit utilise l'API TMDB mais n'est ni approuvé ni certifié par TMDB.
> (*This product uses the TMDB API but is not endorsed or certified by TMDB.*)

---

## 1. Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt                         # pipeline de données (figé)
pip install -r requirements-ml.txt                      # PyTorch (prototype de fusion)
cp .env.example .env                                    # TMDB_READ_TOKEN, seulement pour collecter

dvc pull          # récupère les données versionnées (voir §3)…
dvc repro         # …ou les reconstruit depuis les sources : UNE commande
```

Essai hors ligne en quelques secondes, sans identifiant ni téléchargement (jeu synthétique au format réel) :

```bash
python -m src.pipeline --params configs/params.ci.yaml
```

## 2. Architecture du pipeline

```
sources ──► ingest_movielens ─┬─► ingest_tmdb ──► ingest_posters ─┐
            ingest_kaggle ────┘                                    ├─► validate_raw ─► preprocess_catalog
inbox/  ──► ingest_external (porte de qualité) ────────────────────┼──► preprocess_ratings, align, extract_features
                                                                   │          ┌──────────────┼───────────────┬──────────────┐
                                                                   │   preprocess_text  preprocess_images  preprocess_meta  preprocess_ratings
                                                                   │          └──────────────┼───────────────┘              │
                                                                   │                       align ─────────────────► split ◄─┘
                                                                   │                                                  │
                                                                   └──────► extract_features ◄── validate_processed ◄─┘
                                                                                   │
                                                                                report  (HTML, profil, dérive, data card)
                                                         [Tâche 4 : train ─► evaluate, réservés dans dvc.yaml]
```

`dvc dag` affiche le graphe réel. Chaque étape est un module `python -m src.<paquet>.<module>` aux entrées/sorties
explicites ; DVC ne relance que les étapes dont le code, les données ou **les paramètres déclarés** ont changé
(ex. modifier `preprocessing.text.min_len_chars` ne relance que `preprocess_text` et l'aval).

```
configs/            params.yaml (unique source des chemins, seuils, graines) · params.ci.yaml (échantillon, hérite)
src/common/         configuration, journaux JSON, exécution uniforme des étapes, E/S atomiques
src/ingestion/      movielens · tmdb (API, débit limité, reprise) · posters · kaggle · external (inbox) · synthetic (CI)
src/validation/     checks (porte de qualité) · schemas · raw · processed · demo_fail
src/preprocessing/  catalog · text · image · tabular · ratings · normalize (vocabulaires canoniques)
src/alignment/      appariement movie_id ↔ tmdb_id, masques de modalités
src/splitting/      découpage chronologique + démarrage à froid strict, sans fuite ; gel (freeze) de la spécification
src/features/       représentations par modalité mises en cache (backend light / pretrained)
src/fusion/         prototype PyTorch : GMU masqué à 4 voies, baselines concat / attention
src/training/       suivi d'expériences (MLflow, repli JSON)
src/reporting/      rapport HTML + tableau de bord, profil, dérive (PSI), data card
tests/              62 tests : unitaires, tests de données (défauts injectés), faux serveur TMDB, fusion
dvc.yaml · dvc.lock · .github/workflows/ · Dockerfile · Makefile
```

## 3. Versionnement des données (DVC)

Git versionne le code, `dvc.yaml`, `dvc.lock` (empreintes de chaque donnée) et les rapports légers (`reports/`).
Les données (`data/raw`, `data/interim`, `data/processed`, `data/features`) sont dans le cache DVC et sur un
**remote** ; jamais dans Git.

Remote par défaut : un dossier partagé, `../dvc-storage-movies` (`.dvc/config`).

```bash
dvc push                                           # publier la version courante des données
dvc pull                                           # la récupérer (tiers, autre machine, CI)
git checkout task3 && dvc checkout                 # revenir aux données exactes de l'étiquette task3
```

**Récupération par un tiers.** Cloner le dépôt, puis pointer le remote vers l'emplacement partagé, sans modifier
le fichier versionné (option `--local`) :

```bash
dvc remote modify --local shared url /chemin/vers/dossier/partagé    # dossier réseau / disque commun
# ou Google Drive (dossier partagé du binôme) :
pip install dvc-gdrive
dvc remote add --local -d gdrive gdrive://<id_du_dossier>
dvc pull
```

Sans accès au remote, `dvc repro` reconstruit tout depuis les sources publiques (identifiant TMDB requis pour la
collecte ; le cache TMDB expire après 180 jours, conformément aux conditions de l'API).

## 4. Ajouter ses propres données (inbox)

Un tiers (enseignant, binôme) peut déposer des films et/ou des notes dans [`inbox/`](inbox/README.md) :
`movies.csv` (nouveaux films : titre, année, genres, synopsis, langue, durée, affiche), `ratings.csv` (notes sur des
films MovieLens ou nouveaux, utilisateurs existants ou nouveaux) et `posters/`. Puis `dvc repro` :

- seules `ingest_external` et les étapes en aval sont ré-exécutées ;
- `ingest_external` contrôle les fichiers (colonnes, UTF-8, collision d'identifiants, film inconnu, note hors domaine,
  horodatage, affiche illisible…) et **arrête le pipeline** au moindre défaut critique (`reports/validation/external.json`) ;
- les nouveaux films reçoivent leurs représentations (texte, image, méta) comme les autres : le modèle pourra les
  scorer à partir du seul contenu, sans ré-entraînement ;
- le tableau de bord affiche la source `external` (films, notes, segment).

**Découpage figé.** Avec `split.mode: frozen` (défaut), train/val/test sont définis par
[`configs/split_spec.json`](configs/split_spec.json), versionné : les notes externes vont dans un segment `new` et
l'évaluation reste comparable d'une version des données à l'autre. Pour intégrer les nouvelles données à
l'apprentissage : `python -m src.splitting.freeze` (nouvelle base, à committer) puis `dvc repro`.

```bash
python -m src.ingestion.external --example       # jeu d'exemple valide : 2 films, 5 notes, 2 affiches
dvc repro                                        # -> 2 films « TIM », 5 notes dans le segment new
python -m src.ingestion.external --clear
python -m src.ingestion.external --example-bad   # collision, affiche absente, note à 7★, film inconnu
dvc repro                                        # -> arrêt à ingest_external (code 1)
python -m src.ingestion.external --clear
```

## 5. Qualité des données

Trois portes de qualité **bloquantes** : `validate_raw` (sources), `ingest_external` (inbox) et `validate_processed`
(tables finales). Un contrôle
de sévérité `error` en échec renvoie le code 1 et arrête `dvc repro` ; `warn` est seulement signalé.

| Famille | Exemples de contrôles |
|---|---|
| Schéma et types | colonnes, types, domaines (note ∈ [0,5 ; 5] au pas de 0,5), clés uniques |
| Par modalité | texte : UTF-8 strict, NFC, sans caractère de contrôle ; image : JPEG RGB décodable, largeur 342 px, ratio 1,2–1,8 ; tabulaire : codes ISO 639, multi-hot cohérent |
| Manquants / doublons / aberrants | taux par colonne ≤ seuil, (user, film) unique, année et durée dans des bornes |
| Cohérence inter-modalités | `has_*` cohérents avec le contenu, absence explicite, clés de jointure valides, couverture des 3 modalités |
| Synchronisation temporelle | année de sortie ≤ première note, fraîcheur de l'instantané TMDB |
| Fuite | films froids absents de train/val/test, train < val < test, aucune paire partagée |

Seuils : `validation.*` dans `configs/params.yaml`. Résultats : `reports/validation/*.json` et `reports/quality_report.html`.

**Démonstration d'un échec contrôlé** (les vraies données ne sont jamais modifiées) :

```bash
python -m src.validation.demo_fail                 # note à 7★, film orphelin -> code 1
python -m src.validation.demo_fail --kind leakage  # film froid recopié dans le train -> code 1
```

## 6. Observabilité et traçabilité

- `logs/pipeline.jsonl` : journal structuré (une ligne JSON par événement).
- `reports/stages/<étape>.json` : statut, durée, commit, empreinte des paramètres, volumes et taux de rejet.
- `reports/quality_report.html` : rapport autonome avec tableau de bord des étapes, contrôles, distributions et dérive.
- `reports/data_card.md` : fiche de données (sources, licences, volumes, biais, prétraitements), régénérée à chaque exécution.
- `data/processed/manifest.json` : versions et empreintes des sources, paramètres de découpage.
- Colonnes `overview_source` et `poster_path_source` : provenance de chaque valeur (tmdb ou kaggle).

**Dérive entre deux versions** : `reports/profile.json` contient des histogrammes à bornes fixes ; l'étape `report`
calcule le PSI par rapport à `reports/reference_profile.json`.

```bash
python -m src.reporting.drift --ref-rev task3      # comparer à une version étiquetée
python -m src.reporting.drift --promote            # figer la version courante comme référence
dvc metrics diff task3                             # écarts des statistiques scalaires
```

## 7. Intégration continue

`.github/workflows/ci.yml`, à chaque push : installation des dépendances figées, `ruff`, `pytest`, puis **`dvc repro`
complet sur l'échantillon synthétique** et vérification que la démonstration d'échec échoue. Le rapport HTML est
publié comme artefact. MovieLens et TMDB interdisant la redistribution, la CI n'utilise aucune donnée réelle.

`.github/workflows/scheduled-collect.yml` (bonus) : collecte mensuelle planifiée, `dvc repro`, dérive, `dvc push`.
Il nécessite les secrets `TMDB_READ_TOKEN` et ceux du remote.

## 8. Fusion multimodale (Partie B)

Approche retenue : **fusion intermédiaire par unité à porte (GMU) masquée à quatre voies** (texte, image, méta, et
identifiant collaboratif traité comme une modalité absente pour les films froids), sur des encodeurs pré-entraînés
**figés** dont les représentations sont mises en cache par `extract_features`. Pertes : MSE + têtes unimodales +
alignement contrastif texte ↔ image. Justification, équations et protocole : `docs/Rapport_Tache3.pdf`.

```bash
python -m src.fusion.smoke --fusion gmu            # passe avant + rétropropagation sur un batch réel
pytest tests/test_fusion.py                        # dimensions, modalités absentes, gradients
```

## 9. Commandes utiles

| But | Commande |
|---|---|
| Pipeline complet (périmé seulement) | `dvc repro` (ou `make pipeline`) |
| Étapes impactées par une modification | `dvc status` |
| Une seule étape, sans DVC | `python -m src.preprocessing.text --set preprocessing.text.min_len_chars=30` |
| Déposer / retirer des données externes | `python -m src.ingestion.external --example` · `--clear` |
| Re-geler le découpage d'évaluation | `python -m src.splitting.freeze` (puis commit de `configs/split_spec.json`) |
| Échantillon synthétique | `python -m src.pipeline --params configs/params.ci.yaml` |
| Tests / lint | `pytest -q` · `ruff check src tests` |
| Conteneur | `docker build -t movies-pipeline . && docker run --rm movies-pipeline` |

## 10. Conditions d'utilisation des données

Les conditions d'API de TMDB réservent l'usage commercial à un accord écrit, et l'entraînement d'un modèle de ML sur
du contenu TMDB peut être assimilé à un usage commercial : **à valider avec l'encadrant** avant l'entraînement.
Ne publiez ni les affiches ni les synopsis collectés ; MovieLens interdit aussi la redistribution. Le code applique :
cache TMDB ≤ 180 jours, attribution dans les rapports, clé d'API uniquement par variable d'environnement.
