FROM python:3.13-slim

WORKDIR /app

# Create non-root user for running the service
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser

COPY pyproject.toml ./

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -e .

COPY --chown=10001:10001 qdrant_ingester/ ./qdrant_ingester/

USER 10001:10001

EXPOSE 8002

CMD ["uvicorn", "qdrant_ingester.main:app", "--host", "0.0.0.0", "--port", "8002"]