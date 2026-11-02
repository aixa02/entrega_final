"""Punto de entrada de la API.

Levantar con:  uvicorn app.main:app
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.api.routes import router
from app.core.config import get_settings
from app.core.llm import crear_llm
from app.graph.builder import construir_grafo
from app.observability.tracing import configurar_observabilidad
from app.rag.retriever import get_rag
from app.services.jobs import get_redis
from app.services.worker import task_worker


def crear_checkpointer():
    """Checkpointer de LangGraph en Redis (requiere Redis Stack o Redis 8+)."""
    return AsyncRedisSaver.from_conn_string(get_settings().redis_url)


def activar_detector_de_bloqueos() -> None:
    """Modo debug de asyncio: registra un WARNING por cada tarea que ocupa el
    event loop más de 100 ms (por ejemplo, una llamada sync escondida)."""
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("asyncio").warning("🔍 Detector de bloqueos del event loop activo (umbral: 100 ms)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # valida el .env: si falta algo, la app no arranca
    if settings.async_debug:
        activar_detector_de_bloqueos()
    configurar_observabilidad()

    # Cargar el modelo de embeddings y BM25 es lento y sync: una vez, en un thread.
    await asyncio.to_thread(get_rag)

    async with crear_checkpointer() as checkpointer:
        await checkpointer.asetup()  # crea los índices de checkpoints en Redis
        grafo = construir_grafo(crear_llm(), checkpointer)

        workers = [asyncio.create_task(task_worker(f"worker-{i}", grafo)) for i in range(1, settings.workers + 1)]
        yield
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    await get_redis().aclose()


app = FastAPI(title="Orquestador Multi-Agente", lifespan=lifespan)
app.include_router(router)
