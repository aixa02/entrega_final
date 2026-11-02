"""Estado de los jobs y cola de trabajo, ambos en Redis."""

import json
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
from typing import Any

from redis import asyncio as aioredis

from app.core.config import get_settings


class JobStatus(StrEnum):
    PENDING = "PENDING"                     # en la cola, esperando un worker
    RUNNING = "RUNNING"                     # un worker lo está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"   # pausado por el HITL
    DONE = "DONE"                           # terminó (aprobado o rechazado)
    FAILED = "FAILED"                       # hubo una excepción


# Cuánto espera el worker a que llegue un trabajo en cada BLPOP.
ESPERA_COLA_SEGUNDOS = 5


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    """Cliente async de Redis, compartido por la API y los workers.

    socket_timeout tiene que ser MAYOR que la espera del BLPOP: si fueran iguales
    (el default de redis-py es 5 s), el cliente cortaría la conexión justo cuando
    Redis va a responder "cola vacía" y lo tomaría como TimeoutError.
    """
    return aioredis.from_url(
        get_settings().redis_url,
        decode_responses=True,
        socket_timeout=ESPERA_COLA_SEGUNDOS + 10,
    )


def _clave(job_id: str) -> str:
    return f"{get_settings().job_prefix}{job_id}"


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


async def crear_job(job_id: str, query: str) -> dict[str, Any]:
    job = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "query": query,
        "created_at": _ahora(),
        "updated_at": _ahora(),
    }
    await get_redis().set(_clave(job_id), json.dumps(job))
    return job


async def leer_job(job_id: str) -> dict[str, Any] | None:
    data = await get_redis().get(_clave(job_id))
    return json.loads(data) if data else None


async def actualizar_job(job_id: str, **campos: Any) -> dict[str, Any]:
    job = await leer_job(job_id) or {"job_id": job_id}
    job.update(campos, updated_at=_ahora())
    await get_redis().set(_clave(job_id), json.dumps(job, default=str))
    return job


async def encolar(mensaje: dict[str, Any]) -> None:
    """RPUSH agrega al final de la cola; el worker saca del principio (FIFO)."""
    await get_redis().rpush(get_settings().queue_name, json.dumps(mensaje))


async def desencolar(timeout: int = ESPERA_COLA_SEGUNDOS) -> dict[str, Any] | None:
    """BLPOP espera hasta `timeout` segundos a que haya un mensaje, sin bloquear el event loop."""
    item = await get_redis().blpop(get_settings().queue_name, timeout=timeout)
    return json.loads(item[1]) if item else None
