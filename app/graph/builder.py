"""Construcción del grafo: supervisor + especialistas + síntesis + aprobación humana."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import crear_nodo_analista, crear_nodo_investigador, crear_nodo_sintesis, crear_nodo_supervisor
from app.graph.hitl import enrutar_aprobacion, nodo_aprobacion_humana, nodo_guardar_informe
from app.graph.state import AgentState


def enrutar(state: AgentState) -> Literal["investigador", "analista", "sintesis"]:
    """Traduce la decisión del supervisor al nodo siguiente."""
    if state["next_agent"] == "FINISH":
        return "sintesis"
    return state["next_agent"]


def construir_grafo(llm: BaseChatModel, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    grafo = StateGraph(AgentState)

    grafo.add_node("supervisor", crear_nodo_supervisor(llm))
    grafo.add_node("investigador", crear_nodo_investigador(llm))
    grafo.add_node("analista", crear_nodo_analista(llm))
    grafo.add_node("sintesis", crear_nodo_sintesis(llm))
    grafo.add_node("aprobacion_humana", nodo_aprobacion_humana)
    grafo.add_node("guardar_informe", nodo_guardar_informe)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar)
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("sintesis", "aprobacion_humana")
    grafo.add_conditional_edges("aprobacion_humana", enrutar_aprobacion)
    grafo.add_edge("guardar_informe", END)

    # El checkpointer guarda el estado después de cada nodo, por thread_id (= job_id).
    return grafo.compile(checkpointer=checkpointer)
