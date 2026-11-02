"""Configuración centralizada, validada con Pydantic.

Todo lo que depende del entorno (claves, nombres de modelos, índice de Pinecone,
URL de Redis) se lee del .env. Los campos sin valor por defecto son obligatorios:
si faltan, la app no arranca y el error dice exactamente cuál falta.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]

# Además de validar con Settings, se cargan las variables al entorno del proceso:
# LangSmith (LANGSMITH_*) las lee directamente de ahí.
load_dotenv(RAIZ / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    google_api_key: str = Field(min_length=1)
    llm_model: str
    llm_rpm: float = Field(4, gt=0, description="Pedidos por minuto al LLM (plan gratuito de Gemini: 5)")

    # --- RAG (Pinecone + BM25) ---
    pinecone_api_key: str = Field(min_length=1)
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str
    embedding_dim: int = Field(gt=0)
    rag_top_k: int = Field(5, gt=0)
    docs_dir: Path = RAIZ / "data" / "docs"
    chunks_path: Path = RAIZ / "data" / "chunks.json"

    # --- Orquestador ---
    max_pasos: int = Field(6, gt=0, description="Máximo de intervenciones de especialistas")
    recursion_limit: int = Field(25, gt=0)

    # --- Redis, cola y workers ---
    redis_url: str
    queue_name: str = "tasks_queue"
    job_prefix: str = "job:"
    workers: int = Field(1, gt=0)

    # --- Diagnóstico ---
    async_debug: bool = Field(False, description="Avisa si algo bloquea el event loop más de 100 ms")

    # --- HITL ---
    informes_dir: Path = RAIZ / "informes"


@lru_cache
def get_settings() -> Settings:
    """Se crea una sola vez y se reutiliza en toda la app."""
    return Settings()
