"""Agente de investigación: busca en la documentación de FastAPI (Vector DB de la Pre-entrega 4)."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.core.llm import texto
from app.graph.state import AgentState
from app.rag.retriever import get_rag


async def buscar_fragmentos(pregunta: str) -> list[tuple[str, str]]:
    """Devuelve (fuente, contenido) de los fragmentos más relevantes."""
    return [(d.metadata.get("source", "?"), d.page_content) for d in await get_rag().abuscar(pregunta)]


@tool
async def buscar_documentacion_fastapi(pregunta: str) -> str:
    """Busca en la documentación oficial del tutorial de FastAPI (request body, path y
    query parameters, manejo de errores, CORS, middleware, background tasks,
    dependencias). Usar para cualquier pregunta sobre cómo funciona FastAPI.
    La pregunta debe estar en INGLÉS, porque la documentación está en inglés."""
    resultados = await buscar_fragmentos(pregunta)
    if not resultados:
        return "No se encontró información relevante en la documentación."
    return "\n\n".join(f"[Fuente: {fuente}]\n{contenido}" for fuente, contenido in resultados)


PROMPT_INVESTIGADOR = """Sos un agente de investigación. Tu única función es buscar datos concretos
en la documentación de FastAPI usando tu herramienta.

Reglas:
- Buscá siempre con la herramienta; no respondas de memoria.
- NO valides ni analices datos: eso lo hace otro agente.
- Respondé en español, de forma breve, con los datos encontrados y su fuente (nombre del archivo).
- Si la documentación no tiene el dato, decilo explícitamente."""


def crear_nodo_investigador(llm: BaseChatModel):
    # Herramientas acotadas: este agente SOLO puede buscar en la documentación.
    agente = create_agent(model=llm, tools=[buscar_documentacion_fastapi], system_prompt=PROMPT_INVESTIGADOR)

    async def nodo_investigador(state: AgentState) -> dict:
        # Contexto acotado: el agente recibe solo la instrucción del supervisor,
        # no el historial completo del sistema.
        instruccion = state["instruccion"]
        resultado = await agente.ainvoke({"messages": [HumanMessage(content=instruccion)]})
        respuesta = texto(resultado["messages"][-1])
        paso = state.get("pasos", 0) + 1

        return {
            "messages": [AIMessage(content=respuesta, name="investigador")],
            "contribuciones": [
                {"agente": "investigador", "instruccion": instruccion, "aporte": respuesta, "paso": paso}
            ],
            "pasos": paso,
        }

    return nodo_investigador
