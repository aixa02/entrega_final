"""Worker: saca jobs de la cola de Redis, ejecuta el grafo y actualiza el estado."""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.core.config import get_settings
from app.observability.tracing import metadata_job, traceable
from app.services.jobs import (
    ContribucionResumen,
    Job,
    JobStatus,
    MensajeCola,
    actualizar_job,
    desencolar,
    encolar,
    leer_job,
)

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


def config_grafo(job_id: str, accion: str) -> dict[str, Any]:
    # thread_id = job_id: así el checkpointer sabe qué estado retomar.
    return {
        "configurable": {"thread_id": job_id},
        "recursion_limit": get_settings().recursion_limit,
        "run_name": f"orquestador:{accion}",
        **metadata_job(job_id, accion),
    }


_CONTINUAR = object()  # marcador: retomar desde el último checkpoint


async def _entrada(grafo: CompiledStateGraph, mensaje: MensajeCola, job: Job, config: dict) -> Any:
    """Decide con qué se invoca el grafo. None = no hace falta ejecutar nada."""
    if mensaje.accion == "iniciar":
        return estado_inicial(job.query)
    if mensaje.accion == "reanudar":
        return Command(resume={"aprobado": mensaje.aprobado, "comentario": mensaje.comentario})

    # "continuar": el job se cortó por un reinicio. Se mira qué guardó el checkpointer.
    snapshot = await grafo.aget_state(config)
    if not snapshot.values:
        return estado_inicial(job.query)  # no llegó a guardarse ningún nodo: empieza de cero
    if snapshot.interrupts or not snapshot.next:
        return None  # ya estaba pausado o terminado: solo hay que reflejarlo en el job
    return _CONTINUAR


# process_inputs: en la traza se registra solo el mensaje, no el objeto del grafo.
@traceable(name="procesar_job", run_type="chain", process_inputs=lambda i: {"mensaje": i.get("mensaje")})
async def procesar_job(grafo: CompiledStateGraph, mensaje: MensajeCola) -> Job:
    """Ejecuta, reanuda o retoma el grafo de un job y guarda el resultado."""
    job = await leer_job(mensaje.job_id)
    if job is None:
        raise KeyError(f"No existe el job {mensaje.job_id}")

    config = config_grafo(job.job_id, mensaje.accion)
    entrada = await _entrada(grafo, mensaje, job, config)
    if entrada is _CONTINUAR:
        # Con input None, LangGraph continúa desde el último checkpoint guardado:
        # los nodos que ya habían terminado NO se vuelven a ejecutar.
        await grafo.ainvoke(None, config)
    elif entrada is not None:
        await grafo.ainvoke(entrada, config)

    # El checkpointer dice si el grafo quedó pausado o terminó.
    snapshot = await grafo.aget_state(config)
    if snapshot.interrupts:
        pausa = snapshot.interrupts[0].value
        return await actualizar_job(job.job_id, status=JobStatus.WAITING_APPROVAL, borrador=pausa["borrador"])

    valores = snapshot.values
    aprobacion = valores.get("aprobacion") or {}
    return await actualizar_job(
        job.job_id,
        status=JobStatus.DONE,
        resultado=valores.get("respuesta_final"),
        aprobado=aprobacion.get("aprobado"),
        informe_path=valores.get("informe_path"),
        contribuciones=[ContribucionResumen(paso=c["paso"], agente=c["agente"]) for c in valores.get("contribuciones", [])],
    )


async def task_worker(nombre: str, grafo: CompiledStateGraph) -> None:
    log.warning("%s iniciado", nombre)
    while True:
        try:
            mensaje = await desencolar()
            if mensaje is None:
                continue  # cola vacía: vuelve a esperar
        except asyncio.CancelledError:
            break  # la API se está apagando
        except Exception:
            log.exception("%s: mensaje inválido o error leyendo la cola", nombre)
            await asyncio.sleep(1)
            continue

        job_id = mensaje.job_id
        try:
            await actualizar_job(job_id, status=JobStatus.RUNNING, worker=nombre)
            job = await procesar_job(grafo, mensaje)
            log.warning("%s: job %s → %s", nombre, job_id, job.status)
        except asyncio.CancelledError:
            # Apagado ordenado en medio de un job: no se pierde. Los nodos terminados
            # ya están en el checkpointer; se reencola para retomarlo al reiniciar.
            await actualizar_job(job_id, status=JobStatus.PENDING, worker=None)
            await encolar(MensajeCola(job_id=job_id, accion="continuar"), prioridad=True)
            log.warning("%s: job %s interrumpido, vuelve a la cola", nombre, job_id)
            break
        except Exception as e:
            # Sin esto, el cliente quedaría haciendo polling para siempre en RUNNING.
            log.exception("%s: job %s falló", nombre, job_id)
            await actualizar_job(job_id, status=JobStatus.FAILED, error=f"{type(e).__name__}: {e}")
