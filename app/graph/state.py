"""Esquema del estado compartido entre el supervisor y los especialistas."""

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import MessagesState


class Contribucion(TypedDict):
    """Registro de un aporte: quién lo hizo, qué se le pidió y qué devolvió."""

    agente: str
    instruccion: str
    aporte: str
    paso: int


class Aprobacion(TypedDict):
    """Decisión humana sobre el informe (la completa el nodo HITL)."""

    aprobado: bool
    comentario: Optional[str]


class AgentState(MessagesState):
    """Estado de la Pre-entrega 6 más los campos del flujo de aprobación.

    - contribuciones: operator.add suma cada aporte en vez de pisar la lista.
    - respuesta_final: el informe que arma la síntesis y que revisa el humano.
    - aprobacion: decisión humana (aprobado/rechazado + comentario).
    - informe_path: dónde se guardó el informe, si se aprobó.
    """

    next_agent: Optional[str]
    instruccion: str
    contribuciones: Annotated[list[Contribucion], operator.add]
    pasos: int
    task_completed: bool
    respuesta_final: Optional[str]
    aprobacion: Optional[Aprobacion]
    informe_path: Optional[str]
