# Build context is the repository root (see infrastructure/docker-compose.yml):
#   docker compose --profile api up -d --build rag-api
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch: the default Linux wheel pulls ~2 GB of CUDA libraries the API cannot use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The API does not need the Kafka client used by the ingestion consumer.
COPY requirements.txt .
RUN grep -v '^confluent-kafka' requirements.txt > /tmp/requirements.api.txt \
    && pip install -r /tmp/requirements.api.txt

# rag_api reuses ingestion's embedding service and config, so both packages are copied.
COPY ingestion ./ingestion
COPY rag_api ./rag_api

RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown -R app /models /app
USER app

EXPOSE 8000
CMD ["python", "-m", "rag_api.main"]
