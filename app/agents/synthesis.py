"""Fase de síntesis: combina los aportes en el informe final (que después revisa un humano)."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.core.llm import texto
from app.graph.state import AgentState


def crear_nodo_sintesis(llm: BaseChatModel):
    async def nodo_sintesis(state: AgentState) -> dict:
        pregunta_original = texto(state["messages"][0])
        aportes = "\n\n".join(f"- [{c['agente']}] {c['aporte']}" for c in state.get("contribuciones", []))

        prompt = (
            f"Pregunta original: {pregunta_original}\n\n"
            f"Aportes del equipo:\n{aportes}\n\n"
            "Redactá una respuesta final clara y breve en español combinando estos aportes. "
            "Usá solo la información de los aportes; no agregues datos nuevos. "
            "Mencioná la fuente de la documentación."
        )
        respuesta = texto(await llm.ainvoke([HumanMessage(content=prompt)]))

        return {
            "messages": [AIMessage(content=respuesta, name="sintesis")],
            # Se guarda aparte para que el nodo de aprobación lo muestre como borrador.
            "respuesta_final": respuesta,
            "task_completed": True,
        }

    return nodo_sintesis
