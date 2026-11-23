# Bonus — conteneur reproductible du pipeline (CPU).
#   docker build -t movies-pipeline .
#   docker run --rm movies-pipeline                                    # pipeline sur l'échantillon synthétique
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" --env-file .env movies-pipeline \
#              python -m src.pipeline                                  # vraies données (montées)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN useradd --create-home runner && chown -R runner /app
USER runner

CMD ["python", "-m", "src.pipeline", "--params", "configs/params.ci.yaml"]
