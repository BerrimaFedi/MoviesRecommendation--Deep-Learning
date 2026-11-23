# Raccourcis (GNU make). Sous Windows sans make : les commandes équivalentes sont dans le README.
PY ?= python

.PHONY: install pipeline sample test lint ci demo-fail demo-external demo-external-bad inbox-clear freeze drift baseline dag smoke docker clean-sample

install:
	$(PY) -m pip install -r requirements.txt -r requirements-ml.txt

pipeline:            ## reconstruit les données préparées à partir des sources (étapes périmées seulement)
	dvc repro

sample:              ## pipeline complet sur le jeu synthétique (hors ligne, < 10 s)
	$(PY) -m src.pipeline --params configs/params.ci.yaml

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

ci: lint test sample

demo-fail:           ## échec contrôlé d'un test de qualité (copie corrompue des sources)
	$(PY) -m src.validation.demo_fail

demo-external:       ## dépose un jeu externe valide dans inbox/ puis reconstruit
	$(PY) -m src.ingestion.external --example && dvc repro

demo-external-bad:   ## dépose un jeu externe défectueux : dvc repro doit s'arrêter
	$(PY) -m src.ingestion.external --example-bad && dvc repro

inbox-clear:
	$(PY) -m src.ingestion.external --clear

freeze:              ## re-gèle le découpage d'évaluation sur les données actuelles
	$(PY) -m src.splitting.freeze

drift:               ## dérive par rapport à la référence figée
	$(PY) -m src.reporting.drift

baseline:            ## le profil courant devient la référence de dérive
	$(PY) -m src.reporting.drift --promote

dag:
	dvc dag

smoke:               ## passe avant du prototype de fusion sur un batch réel
	$(PY) -m src.fusion.smoke

docker:
	docker build -t movies-pipeline . && docker run --rm movies-pipeline

clean-sample:
	rm -rf data_sample reports_sample
