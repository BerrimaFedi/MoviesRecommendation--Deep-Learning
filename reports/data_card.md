# Fiche de données — Projet 10, recommandation multimodale de films

> Fichier généré par l'étape `report` (`src/reporting/datacard.py`) — ne pas éditer à la main.
> Mode d'ingestion : **remote**.

## 1. Motivation et usage prévu

Recommander des films à des utilisateurs MovieLens, y compris des films **jamais notés** (démarrage à froid),
à partir de trois modalités de contenu : affiche (image), synopsis (texte), métadonnées (tabulaire).
Usage strictement académique. **Hors périmètre** : usage commercial, redistribution des affiches et synopsis.

## 2. Sources, licences et conditions

| Source | Contenu | Licence / conditions | Version |
|---|---|---|---|
| MovieLens (GroupLens) | notes, catalogue, liens IMDb/TMDB | usage de recherche, redistribution interdite, citation requise (Harper & Konstan, 2015) | `ml-latest-small` |
| API TMDB | synopsis, affiches w342, durée, langue, genres, popularité | API Terms of Use : attribution obligatoire, cache ≤ 6 mois, usage commercial soumis à accord — l'entraînement de modèles ML peut être assimilé à un usage commercial, à valider avec l'encadrant | langue `en-US`, taille `w342` |
| Kaggle « The Movies Dataset » | secours : synopsis, durée, chemin d'affiche | licence du dépôt Kaggle (CC0 déclarée) ; instantané TMDB de juillet 2017, pas une source indépendante | `rounakbanik/the-movies-dataset` |

*This product uses the TMDB API but is not endorsed or certified by TMDB.*

## 3. Composition (version courante)

- **9,742 films**, **100,836 notes**, **610 utilisateurs** — densité 1.7 %.
- Couverture : synopsis 99.5 % · affiche 99.0 % · métadonnées 99.5 % ·
  les trois 99.0 %. Durée manquante 0.5 %, année manquante 0.1 %.
- Synopsis : médiane de 43 mots. Affiches : 342 px de large, ratio ≈ 1,5.
- Notes : moyenne 3.50 sur une échelle 0,5–5 au pas de 0,5. Médiane de
  3 note(s) par film ; 66.4 % des films ont ≤ 5 notes.
- Langues : en 84.8 %, fr 3.5 %, ja 2.7 %, it 1.5 %. Genres les plus fréquents : drama 52.8 %, comedy 42.8 %, thriller 25.3 %, romance 23.2 %, action 22.8 %.
- Découpage : cold_item_test 4.5 %, test 9.6 %, train 76.4 %, val 9.6 %.

## 4. Prétraitements (une étape DVC chacun)

1. `preprocess_catalog` — genres MovieLens → vocabulaire canonique de 20 genres (Children→Family, Sci-Fi→Science
   Fiction, IMAX → indicateur `is_imax`), année extraite du titre, identifiant IMDb normalisé.
2. `preprocess_text` — Unicode NFC, caractères de contrôle supprimés, espaces
   normalisés ; synopsis < 20 caractères = absent ; secours Kaggle.
3. `preprocess_images` — seules les affiches valides (JPEG RGB décodable, côté ≥
   16 px) sont retenues ; empreinte SHA-1 conservée.
4. `preprocess_meta` — langue ISO 639-1 (cn→zh, xx→und), durée 0 = inconnue, popularité/votes TMDB seulement.
5. `preprocess_ratings` — doublons (utilisateur, film) supprimés, notes hors domaine rejetées.
6. `align` — jointure movie_id ↔ tmdb_id ; aucun film supprimé, absences marquées (`has_*`, `modality_mask`).
7. `split` — 5.0 % des films en démarrage à froid strict, puis découpage chronologique
   80%/10%/10%,
   graine 42.

Contrôles de qualité de cette version : raw : 45/45; external : 0/0; processed : 168/168.

## 5. Biais et limites connus

- **Population** : 610 utilisateurs MovieLens volontaires, non représentatifs du grand public ;
  pour latest-small, notes de 1996 à 2018.
- **Popularité** : distribution à longue traîne ; les films très notés dominent l'apprentissage collaboratif.
- **Langue et culture** : catalogue majoritairement anglophone et américain ; synopsis tous en anglais (`en-US`).
- **Fuite temporelle** : popularité et votes TMDB sont des instantanés *actuels*, postérieurs aux notes : exclus des
  représentations du modèle.
- **Genres** : accord MovieLens/TMDB moyen (Jaccard 0.71) ; l'union des deux est utilisée.
- **Affiches** : une affiche reflète le marketing d'une époque et d'un marché ; elle peut changer dans le temps sur TMDB.

## 6. Maintenance

Reconstruction : `dvc repro` (ou `dvc pull` pour récupérer les données versionnées). Cache TMDB à rafraîchir tous les
6 mois au plus (`ingest_tmdb`). Contact : binôme du projet (voir README).
