# Python 3.12, como declara pyproject.toml
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# PyTorch solo-CPU: sentence-transformers instala por defecto la versión con CUDA,
# que pesa varios GB y acá no se usa (los embeddings corren en CPU).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencias primero: Docker reutiliza esta capa si solo cambia el código.
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

# Prepara Pinecone si hace falta (idempotente) y levanta la API.
CMD ["sh", "-c", "python -m scripts.bootstrap && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
