FROM python:3.11-slim@sha256:6d0b2a4a4f7a9e2bde3b3a6d8c8b6f3a1a6d7c4e5f8a0b9c1d2e3f4a5b6c7d8

WORKDIR /app

# Create non-root user for running the service
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY --chown=10001:10001 qdrant_ingester/ ./qdrant_ingester/

USER 10001:10001

EXPOSE 8002

CMD ["uvicorn", "qdrant_ingester.main:app", "--host", "0.0.0.0", "--port", "8002"]