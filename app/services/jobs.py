"""Estado de los jobs y cola de trabajo en Redis, con contratos Pydantic.

Todo lo que entra y sale de Redis pasa por un modelo: si un dato guardado está
corrupto o un mensaje de la cola tiene un formato inesperado, falla la validación
en vez de propagarse por el sistema.
"""

import logging
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from redis import asyncio as aioredis

from app.core.config import get_settings

log = logging.getLogger("jobs")


class JobStatus(StrEnum):
    PENDING = "PENDING"                     # en la cola, esperando un worker
    RUNNING = "RUNNING"                     # un worker lo está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"   # pausado por el HITL
    DONE = "DONE"                           # terminó (aprobado o rechazado)
    FAILED = "FAILED"                       # hubo una excepción


class ContribucionResumen(BaseModel):
    paso: int
    agente: str


class Job(BaseModel):
    """Estado de un job tal como se guarda en Redis y se devuelve en GET /tasks/{id}."""

    job_id: str
    status: JobStatus
    query: str
    created_at: datetime
    updated_at: datetime
    worker: str | None = None
    borrador: str | None = Field(None, description="Informe pendiente de aprobación (WAITING_APPROVAL)")
    resultado: str | None = Field(None, description="Respuesta final (DONE)")
    aprobado: bool | None = None
    informe_path: str | None = None
    contribuciones: list[ContribucionResumen] = []
    error: str | None = Field(None, description="Motivo del fallo (FAILED)")


class MensajeCola(BaseModel):
    """Mensaje de la cola de trabajo.

    - iniciar: ejecutar el grafo desde el principio.
    - reanudar: continuar después de la aprobación humana.
    - continuar: retomar desde el último checkpoint (job interrumpido por un reinicio).
    """

    job_id: str
    accion: Literal["iniciar", "reanudar", "continuar"]
    aprobado: bool | None = None
    comentario: str | None = None


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


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


async def _guardar(job: Job) -> Job:
    await get_redis().set(_clave(job.job_id), job.model_dump_json())
    return job


async def crear_job(job_id: str, query: str) -> Job:
    ahora = _ahora()
    return await _guardar(Job(job_id=job_id, status=JobStatus.PENDING, query=query, created_at=ahora, updated_at=ahora))


async def leer_job(job_id: str) -> Job | None:
    data = await get_redis().get(_clave(job_id))
    return Job.model_validate_json(data) if data else None


async def actualizar_job(job_id: str, **campos: Any) -> Job:
    """Aplica cambios y los VALIDA: un estado inválido no llega a guardarse."""
    job = await leer_job(job_id)
    if job is None:
        raise KeyError(f"No existe el job {job_id}")
    return await _guardar(Job.model_validate({**job.model_dump(), **campos, "updated_at": _ahora()}))


async def encolar(mensaje: MensajeCola) -> None:
    """RPUSH agrega al final de la cola; el worker saca del principio (FIFO)."""
    await get_redis().rpush(get_settings().queue_name, mensaje.model_dump_json())


async def desencolar(timeout: int = ESPERA_COLA_SEGUNDOS) -> MensajeCola | None:
    """BLPOP espera hasta `timeout` segundos a que haya un mensaje, sin bloquear el event loop."""
    item = await get_redis().blpop(get_settings().queue_name, timeout=timeout)
    return MensajeCola.model_validate_json(item[1]) if item else None


async def recuperar_jobs_interrumpidos() -> int:
    """Al arrancar: los jobs que quedaron en RUNNING son de un proceso que se cortó
    de golpe (sin apagado ordenado). Se vuelven a encolar para que retomen desde
    su último checkpoint."""
    recuperados = 0
    async for clave in get_redis().scan_iter(match=f"{get_settings().job_prefix}*"):
        data = await get_redis().get(clave)
        if not data:
            continue
        job = Job.model_validate_json(data)
        if job.status == JobStatus.RUNNING:
            await actualizar_job(job.job_id, status=JobStatus.PENDING, worker=None)
            await encolar(MensajeCola(job_id=job.job_id, accion="continuar"))
            recuperados += 1
    if recuperados:
        log.warning("♻️ %s job(s) interrumpidos vuelven a la cola", recuperados)
    return recuperados
