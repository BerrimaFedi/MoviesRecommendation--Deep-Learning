# Rapport qualité des données

Généré le 2026-11-23T15:59:28+00:00 · mode `remote` · MovieLens `ml-latest-small`

## Volumes

- Films : 9742 · notes : 100836 · utilisateurs : 610 · densité : 1.7 %

## Couverture des modalités

- Synopsis : 99.5 % · affiche : 99.0 % · méta : 99.5 % · trois modalités : 99.0 %

## Contrôles

- `raw` : 45/45 réussis, 0 erreur(s), 0 avertissement(s)
- `external` : 0/0 réussis, 0 erreur(s), 0 avertissement(s)
- `processed` : 168/168 réussis, 0 erreur(s), 0 avertissement(s)

## Dérive

- ok

## Avertissements

- popularity, vote_average et vote_count sont des instantanés TMDB à la date tmdb_fetched_at : ils reflètent l'attention actuelle, pas celle de la période des notes (fuite temporelle possible). Exclus des représentations méta (src/features/extract.py).
- Le dataset Kaggle est un instantané TMDB de 2017 dont les notes viennent de MovieLens : ce n'est pas une source indépendante.
- This product uses the TMDB API but is not endorsed or certified by TMDB.
