FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8002

CMD ["uvicorn", "qdrant_ingester.main:app", "--host", "0.0.0.0", "--port", "8002"]