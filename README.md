# Projet 10 — Collecte des données (tâche 2)

Pipeline reprenable qui construit, à partir de MovieLens, de TMDB et (en secours) de Kaggle, une table de films
multimodale (affiche, synopsis, métadonnées) et une table de notes déjà découpée.

> Ce produit utilise l'API TMDB mais n'est ni approuvé ni certifié par TMDB.
> (*This product uses the TMDB API but is not endorsed or certified by TMDB.*)

## ⚠️ À lire avant de lancer la collecte

**Conditions d'utilisation de TMDB.** Les conditions d'API de TMDB réservent l'usage commercial à un accord écrit,
et plusieurs sources indiquent que l'entraînement ou la validation d'un modèle de ML/IA sur du contenu TMDB est
classé parmi les usages commerciaux. Le cas d'un projet académique non commercial a été posé sur leur forum sans
réponse claire que nous ayons pu lire. **Lisez la clause correspondante des API Terms of Use et validez avec votre
encadrant** (ou écrivez à TMDB) avant d'entraîner un modèle sur ces données. Ne publiez ni les affiches ni les
synopsis collectés ; MovieLens interdit lui aussi la redistribution.

Autres règles appliquées par le code : cache TMDB limité à 6 mois (`cache_max_age_days=180`), attribution TMDB
dans le rapport et le manifeste, clé d'API uniquement via variable d'environnement.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # puis renseigner TMDB_READ_TOKEN (ou TMDB_API_KEY)
```

Identifiants : compte TMDB → Paramètres → API. Utilisez de préférence le « API Read Access Token » (v4, en-tête
`Authorization: Bearer`). Le fichier `.env` est ignoré par Git.

## Utilisation

```bash
# Développement (≈ 9 700 films)
python -m collecte run-all --limit 200          # essai rapide sur 200 films
python -m collecte run-all                      # tout le jeu latest-small

# Résultats finaux (versions stables de MovieLens)
python -m collecte --movielens ml-25m run-all   # ou ml-32m

# Étape par étape
python -m collecte download-movielens
python -m collecte download-kaggle              # facultatif, CLI Kaggle + identifiants requis
python -m collecte fetch-tmdb [--limit N] [--rate 20] [--language en-US]
python -m collecte fetch-posters
python -m collecte build [--no-kaggle]
```

Toutes les étapes sont **reprenables** : relancer une commande ne refait que ce qui manque. Un film déjà collecté
n'est re-demandé que si son cache a plus de 180 jours.

Ordres de grandeur (estimations, non mesurées ici) : `latest-small` ≈ 9 700 requêtes API, soit environ 8 minutes à
20 req/s ; `ml-25m` ≈ 62 000 films, soit environ 50 minutes, et quelques Go d'affiches en w342.

## Ce que fait chaque étape

| Commande | Rôle | Sortie |
|---|---|---|
| `download-movielens` | Télécharge l'archive, vérifie le MD5 si GroupLens le publie, extrait | `data/raw/<version>/` |
| `fetch-tmdb` | `GET /movie/{id}` en parallèle, limitation de débit, reprise sur 429/5xx, 404 mémorisés | `data/raw/tmdb/movies.jsonl` |
| `fetch-posters` | Télécharge l'affiche (w342), vérifie que c'est une vraie image (Pillow), enregistre en JPEG RGB | `data/raw/posters/*.jpg` + `manifest.jsonl` |
| `download-kaggle` | Récupère `movies_metadata.csv` (secours hors ligne) | `data/raw/kaggle/` |
| `build` | Fusionne, normalise genres/langues/dates, calcule les indicateurs, découpe les notes, écrit le rapport | `data/processed/`, `data/reports/` |

## Tables produites (`data/processed/`)

`movies.parquet` — une ligne par film MovieLens (61 colonnes) :

- identifiants : `movie_id`, `tmdb_id`, `imdb_tt`
- texte : `overview` (+ `overview_source` : tmdb ou kaggle), `tagline`, `title`
- image : `poster_file` (chemin relatif à `data/raw/`), `poster_w`, `poster_h`, `poster_sha1`
- métadonnées : `release_year`, `original_language`, `lang_cat` (15 langues + `other` + `und`), `runtime_min`,
  `popularity`, `vote_average`, `vote_count` (+ versions `log_`), `genres_unified` et 20 colonnes `g_*` (multi-hot)
- indicateurs : `has_overview`, `has_poster`, `has_meta`, `n_modalities`, `all_modalities`, `runtime_missing`,
  `year_missing`, `snapshot_features_missing`

`ratings.parquet` — `user_id, movie_id, rating, timestamp, split, user_in_train, item_in_train`, avec
`split ∈ {train, val, test, cold_item_test}` :

- **Découpage chronologique global** : 80 % les plus anciennes notes en `train`, 10 % en `val`, 10 % en `test`
  (les égalités de timestamp restent du même côté, donc pas de fuite).
- **Démarrage à froid strict** : 5 % des films (au moins 5 notes et au moins un contenu) sont retirés de
  `train/val/test`. Toutes leurs notes vont dans `cold_item_test` : le modèle ne les a jamais vues et doit scorer
  ces films avec leur seul contenu. Graine fixée (`seed=42`).
- `item_in_train` / `user_in_train` permettent de séparer, dans `val/test`, les films ou utilisateurs déjà vus
  de ceux qui sont nouveaux.

## Choix de conception (cohérence des sources)

- **Genres** : vocabulaire canonique de 20 genres. `Children→Family`, `Musical→Music`, `Sci-Fi→Science Fiction` ;
  `IMAX` (format, pas genre) devient l'indicateur `is_imax` ; `(no genres listed)` et `Foreign` sont ignorés. Le
  genre final est l'**union** MovieLens ∪ TMDB, et le Jaccard entre les deux sources est publié dans le rapport.
  Tout genre hors vocabulaire est compté, pas perdu en silence.
- **Langues** : code ISO 639-1 en minuscules ; `cn→zh`, `iw→he`, `xx→und`.
- **Synopsis** : une seule langue demandée à TMDB (`en-US`) pour toute la base. En dessous de 20 caractères, le
  synopsis est considéré comme absent.
- **Films incomplets** : jamais supprimés à la collecte, seulement signalés (`has_*`, `n_modalities`). Le choix de
  filtrer ou de masquer une modalité se fait à l'entraînement.
- **Popularité et votes** : instantanés TMDB datés (`tmdb_fetched_at`), non mélangés avec l'instantané Kaggle de
  2017. Ils reflètent l'attention *actuelle* et risquent de fuir l'information dans un découpage chronologique :
  à traiter comme variables optionnelles.
- **Kaggle** : source de secours (synopsis, durée, genres, chemin d'affiche), pas source indépendante : ses notes
  viennent de MovieLens et ses métadonnées sont un instantané TMDB de 2017.

## Tests

```bash
python -m pytest -q
```

23 tests, sans accès réseau : un faux serveur TMDB local reproduit les cas difficiles (429 puis succès, 500 puis
succès, 404, genre inconnu, synopsis vide, affiche absente / en 404 / corrompue, durée 0, codes de langue
atypiques, fichier Kaggle malformé) et vérifie la reprise, l'absence de double téléchargement et l'absence de fuite
entre les découpages.

**Ce qui n'a pas été testé** : l'exécution contre l'API TMDB réelle et contre les serveurs de GroupLens/Kaggle
(l'environnement de développement n'y avait pas accès). Faites d'abord `run-all --limit 200` sur votre machine.
