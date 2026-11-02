"""Nodos de Human-in-the-loop: pausa antes de la acción crítica (guardar el informe)."""

import asyncio
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

from app.core.config import get_settings
from app.core.llm import texto
from app.graph.state import AgentState


async def nodo_aprobacion_humana(state: AgentState) -> dict:
    """Pausa obligatoria antes de guardar el informe.

    interrupt() detiene el grafo y deja el estado en el checkpointer (Redis).
    Al reanudar con Command(resume=...), el nodo se vuelve a ejecutar desde el
    principio e interrupt() devuelve la decisión humana. Por eso no hay nada
    con efectos secundarios ANTES de esta línea.
    """
    decision = interrupt(
        {
            "accion": "guardar_informe",
            "mensaje": "Revisá el borrador y aprobá o rechazá su guardado.",
            "borrador": state.get("respuesta_final", ""),
        }
    )
    return {"aprobacion": {"aprobado": bool(decision.get("aprobado")), "comentario": decision.get("comentario")}}


def enrutar_aprobacion(state: AgentState) -> Literal["guardar_informe", "__end__"]:
    aprobacion = state.get("aprobacion") or {}
    return "guardar_informe" if aprobacion.get("aprobado") else END


def _escribir_informe(nombre: str, contenido: str) -> str:
    directorio = get_settings().informes_dir
    directorio.mkdir(exist_ok=True)
    ruta = directorio / nombre
    ruta.write_text(contenido, encoding="utf-8")
    return str(ruta)


async def nodo_guardar_informe(state: AgentState, config: RunnableConfig) -> dict:
    """Acción crítica con efecto secundario: escribe el informe en disco.

    Solo se llega acá si un humano aprobó. La escritura de archivos es sync,
    así que va a un thread para no bloquear el event loop.
    """
    job_id = config["configurable"]["thread_id"]
    aprobacion = state.get("aprobacion") or {}
    contenido = (
        f"# Informe {job_id}\n\n"
        f"**Consulta:** {texto(state['messages'][0])}\n\n"
        f"{state.get('respuesta_final', '')}\n\n"
        f"---\nAprobado por revisión humana. Comentario: {aprobacion.get('comentario') or '(sin comentario)'}\n"
    )
    ruta = await asyncio.to_thread(_escribir_informe, f"{job_id}.md", contenido)
    return {"informe_path": ruta}
