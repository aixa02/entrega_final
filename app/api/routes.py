"""Endpoints de la API. Ninguno espera al agente: solo leen/escriben en Redis."""

import uuid

from fastapi import APIRouter, HTTPException

from app.api.schemas import AprobacionRequest, TaskRequest
from app.services.jobs import JobStatus, actualizar_job, crear_job, encolar, get_redis, leer_job

router = APIRouter()


@router.post("/tasks", status_code=202)
async def crear_tarea(request: TaskRequest):
    """Encola la tarea y devuelve el job_id al instante."""
    job_id = str(uuid.uuid4())
    await crear_job(job_id, request.query)  # primero el estado, después la cola
    await encolar({"job_id": job_id, "accion": "iniciar"})
    return {"job_id": job_id, "status": JobStatus.PENDING}


@router.get("/tasks/{job_id}")
async def estado_tarea(job_id: str):
    job = await leer_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    return job


@router.post("/tasks/{job_id}/approve", status_code=202)
async def aprobar_tarea(job_id: str, body: AprobacionRequest):
    """Aprueba o rechaza el informe de un job pausado por el HITL."""
    job = await leer_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id no encontrado")
    if job["status"] != JobStatus.WAITING_APPROVAL:
        raise HTTPException(status_code=409, detail=f"El job está en {job['status']}, no espera aprobación")

    # No se reanuda acá: se encola, igual que una tarea nueva, para no bloquear la API.
    await actualizar_job(job_id, status=JobStatus.PENDING, aprobado=body.aprobado)
    await encolar({"job_id": job_id, "accion": "reanudar", "aprobado": body.aprobado, "comentario": body.comentario})
    return {"job_id": job_id, "status": JobStatus.PENDING, "aprobado": body.aprobado}


@router.get("/health")
async def health():
    await get_redis().ping()
    return {"status": "ok"}
