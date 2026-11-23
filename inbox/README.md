# Inbox — ajouter vos propres données

Déposez ici des films et/ou des notes, puis lancez `dvc repro`. Seules l'étape `ingest_external` et l'aval sont
ré-exécutés ; les contrôles de qualité (`reports/validation/external.json`) **arrêtent le pipeline** si un fichier
est invalide. Le tableau de bord (`reports/quality_report.html`) montre ensuite la source « external ».
Les CSV et affiches déposés ici sont ignorés par Git.

```
inbox/
├── movies.csv       (facultatif) nouveaux films
├── ratings.csv      (facultatif) nouvelles notes
└── posters/         (facultatif) affiches référencées par movies.csv
```

## `movies.csv`

| Colonne | Obligatoire | Format |
|---|---|---|
| `movie_id` | oui | entier **nouveau** (absent de MovieLens, ex. ≥ 900000) |
| `title` | oui | texte UTF-8 |
| `year` | non | année de sortie |
| `genres` | non | séparés par `\|` : noms MovieLens (`Sci-Fi`, `Children`…) ou canoniques (`Science Fiction`, `Family`…) |
| `overview` | non | synopsis (≥ 20 caractères pour compter comme modalité texte) |
| `original_language` | non | code ISO 639-1 (`en`, `fr`…) |
| `runtime_min` | non | durée en minutes (1 à 600) |
| `poster_file` | non | nom d'un fichier de `posters/` (JPEG ou PNG) |

## `ratings.csv`

| Colonne | Format |
|---|---|
| `user_id` | entier ; un identifiant MovieLens existant ou un nouvel utilisateur |
| `movie_id` | film MovieLens **ou** film de `movies.csv` |
| `rating` | 0,5 à 5 au pas de 0,5 |
| `timestamp` | secondes Unix ou date ISO (`2025-06-02T10:00:00Z`) |

## Effet sur le découpage

Avec `split.mode: frozen` (défaut), le découpage d'évaluation est figé dans `configs/split_spec.json` : vos notes
vont dans un segment **`new`** et train/val/test restent identiques, donc les métriques restent comparables. Les
nouveaux films sont des films froids : le modèle les score à partir de leur seul contenu (affiche, synopsis,
métadonnées), sans ré-entraînement. Pour les intégrer à l'entraînement : `python -m src.splitting.freeze`, puis
`dvc repro` (nouvelle base d'évaluation, à committer).

## Essayer

```bash
python -m src.ingestion.external --example       # jeu d'exemple valide (2 films, 5 notes, 2 affiches)
dvc repro
python -m src.ingestion.external --clear
python -m src.ingestion.external --example-bad   # jeu défectueux : dvc repro s'arrête à ingest_external
```
