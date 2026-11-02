"""Worker: saca jobs de la cola de Redis, ejecuta el grafo y actualiza el estado."""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.config import get_settings
from app.observability.tracing import metadata_job, traceable
from app.services.jobs import JobStatus, actualizar_job, desencolar, leer_job

log = logging.getLogger("worker")


def estado_inicial(consulta: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=consulta)],
        "next_agent": None,
        "instruccion": "",
        "contribuciones": [],
        "pasos": 0,
        "task_completed": False,
        "respuesta_final": None,
        "aprobacion": None,
        "informe_path": None,
    }


# process_inputs: en la traza se registra solo el mensaje, no el objeto del grafo.
@traceable(name="procesar_job", run_type="chain", process_inputs=lambda i: {"mensaje": i.get("mensaje")})
async def procesar_job(grafo: CompiledStateGraph, mensaje: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta (o reanuda) el grafo para un job. Decorada con @traceable para que
    cada job aparezca en LangSmith como una traza con todos los nodos adentro."""
    job_id = mensaje["job_id"]
    accion = mensaje["accion"]

    # thread_id = job_id: así el checkpointer sabe qué estado retomar al reanudar.
    config = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": get_settings().recursion_limit,
        "run_name": f"orquestador:{accion}",
        **metadata_job(job_id, accion),
    }

    if accion == "iniciar":
        job = await leer_job(job_id)
        entrada: Any = estado_inicial(job["query"])
    else:  # reanudar después de la aprobación humana
        entrada = Command(resume={"aprobado": mensaje["aprobado"], "comentario": mensaje.get("comentario")})

    await grafo.ainvoke(entrada, config)

    # Después de ejecutar, el checkpointer dice si el grafo quedó pausado o terminó.
    snapshot = await grafo.aget_state(config)
    if snapshot.interrupts:
        pausa = snapshot.interrupts[0].value
        return await actualizar_job(job_id, status=JobStatus.WAITING_APPROVAL, borrador=pausa["borrador"])

    valores = snapshot.values
    aprobacion = valores.get("aprobacion") or {}
    return await actualizar_job(
        job_id,
        status=JobStatus.DONE,
        resultado=valores.get("respuesta_final"),
        aprobado=aprobacion.get("aprobado"),
        informe_path=valores.get("informe_path"),
        contribuciones=[{"paso": c["paso"], "agente": c["agente"]} for c in valores.get("contribuciones", [])],
    )


async def task_worker(nombre: str, grafo: CompiledStateGraph) -> None:
    log.warning("%s iniciado", nombre)
    while True:
        try:
            mensaje = await desencolar(timeout=5)
            if mensaje is None:
                continue  # cola vacía: vuelve a esperar
        except asyncio.CancelledError:
            break  # la API se está apagando
        except Exception:
            log.exception("%s: error leyendo la cola", nombre)
            await asyncio.sleep(1)
            continue

        job_id = mensaje["job_id"]
        try:
            await actualizar_job(job_id, status=JobStatus.RUNNING, worker=nombre)
            job = await procesar_job(grafo, mensaje)
            log.warning("%s: job %s → %s", nombre, job_id, job["status"])
        except asyncio.CancelledError:
            await actualizar_job(job_id, status=JobStatus.FAILED, error="La API se apagó durante la ejecución")
            break
        except Exception as e:
            # Sin esto, el cliente quedaría haciendo polling para siempre en RUNNING.
            log.exception("%s: job %s falló", nombre, job_id)
            await actualizar_job(job_id, status=JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
