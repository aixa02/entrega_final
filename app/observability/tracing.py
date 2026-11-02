"""Observabilidad con LangSmith.

LangChain y LangGraph envían trazas solos cuando están estas variables de entorno:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=...
Cada nodo, llamada al LLM, herramienta y retriever queda como un run dentro de la traza.

Para funciones propias (que no son de LangChain) se usa el decorador @traceable,
así también aparecen en el árbol de la traza.
"""

import logging
import os

from langsmith import traceable

log = logging.getLogger("observability")

__all__ = ["configurar_observabilidad", "traceable", "metadata_job"]


def configurar_observabilidad() -> bool:
    activo = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if activo and not os.getenv("LANGSMITH_API_KEY"):
        log.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: no se enviarán trazas.")
        return False
    if activo:
        log.warning("🔭 Trazas activas en LangSmith, proyecto '%s'", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        log.warning("Trazas desactivadas (LANGSMITH_TRACING != true).")
    return activo


def metadata_job(job_id: str, accion: str) -> dict:
    """Metadata y tags que se adjuntan a cada ejecución, para filtrar en el dashboard."""
    return {"metadata": {"job_id": job_id, "accion": accion}, "tags": ["api", accion]}
