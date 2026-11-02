"""Endpoints de la API. Ninguno espera al agente: solo leen o escriben en Redis."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AprobacionRequest,
    AprobacionResponse,
    ContribucionDetalle,
    ConversacionResponse,
    HealthResponse,
    MensajeConversacion,
    TaskCreatedResponse,
    TaskRequest,
)
from app.core.llm import texto
from app.services.jobs import Job, JobStatus, MensajeCola, actualizar_job, crear_job, encolar, get_redis, leer_job

router = APIRouter()


async def _job_o_404(job_id: str) -> Job:
    job = await leer_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    return job


@router.post("/tasks", status_code=202, response_model=TaskCreatedResponse)
async def crear_tarea(request: TaskRequest) -> TaskCreatedResponse:
    """Encola la tarea y devuelve el job_id al instante."""
    job_id = str(uuid.uuid4())
    await crear_job(job_id, request.query)  # primero el estado, después la cola
    await encolar(MensajeCola(job_id=job_id, accion="iniciar"))
    return TaskCreatedResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/tasks/{job_id}", response_model=Job)
async def estado_tarea(job_id: str) -> Job:
    return await _job_o_404(job_id)


@router.get("/tasks/{job_id}/conversation", response_model=ConversacionResponse)
async def conversacion(job_id: str, request: Request) -> ConversacionResponse:
    """Lee el estado del grafo directamente del checkpointer.

    Sirve para comprobar la persistencia: después de reiniciar el servidor,
    la conversación sigue disponible acá.
    """
    await _job_o_404(job_id)
    grafo = request.app.state.grafo
    config = {"configurable": {"thread_id": job_id}}

    snapshot = await grafo.aget_state(config)
    valores = snapshot.values or {}
    checkpoints = [c async for c in grafo.aget_state_history(config)]

    return ConversacionResponse(
        job_id=job_id,
        mensajes=[
            MensajeConversacion(autor=getattr(m, "name", None) or "usuario", contenido=texto(m))
            for m in valores.get("messages", [])
        ],
        contribuciones=[ContribucionDetalle(**c) for c in valores.get("contribuciones", [])],
        siguiente_nodo=list(snapshot.next),
        checkpoints=len(checkpoints),
    )


@router.post("/tasks/{job_id}/approve", status_code=202, response_model=AprobacionResponse)
async def aprobar_tarea(job_id: str, body: AprobacionRequest) -> AprobacionResponse:
    """Aprueba o rechaza el informe de un job pausado por el HITL."""
    job = await _job_o_404(job_id)
    if job.status != JobStatus.WAITING_APPROVAL:
        raise HTTPException(status_code=409, detail=f"El job está en {job.status}, no espera aprobación")

    # No se reanuda acá: se encola para no bloquear la API, pero al principio de la cola.
    await actualizar_job(job_id, status=JobStatus.PENDING, aprobado=body.aprobado)
    await encolar(
        MensajeCola(job_id=job_id, accion="reanudar", aprobado=body.aprobado, comentario=body.comentario),
        prioridad=True,  # no espera detrás de tareas nuevas
    )
    return AprobacionResponse(job_id=job_id, status=JobStatus.PENDING, aprobado=body.aprobado)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    await get_redis().ping()
    return HealthResponse(status="ok")
