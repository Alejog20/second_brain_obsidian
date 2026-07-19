# Second Brain — worker image
#
# Invoked on-demand by launchd, e.g.:
#   docker compose run --rm worker python -m src.jobs.nightly_run
#
# Ollama is NOT installed here: on Apple Silicon, Docker Desktop cannot pass the GPU
# through to a Linux container, so Ollama runs natively on the host (brew install ollama)
# and is reached at http://host.docker.internal:11434 (see docker-compose.yml).

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

CMD ["python", "-m", "src.jobs.nightly_run"]
