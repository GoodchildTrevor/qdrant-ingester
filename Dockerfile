FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY qdrant_ingester/ ./qdrant_ingester/

EXPOSE 8002

CMD ["uvicorn", "qdrant_ingester.main:app", "--host", "0.0.0.0", "--port", "8002"]
