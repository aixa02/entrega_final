"""Agente de análisis: valida datos JSON contra un esquema (validación de esquemas)."""

import json

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import ValidationError, create_model

from app.core.llm import texto
from app.graph.state import AgentState

# Tipos permitidos en el esquema. Se mapean a mano (sin eval) para que el LLM
# no pueda inyectar código: solo puede elegir uno de estos nombres.
TIPOS = {"str": str, "int": int, "float": float, "bool": bool}


@tool
async def validar_json_contra_esquema(esquema_json: str, datos_json: str) -> str:
    """Valida si un objeto JSON cumple un esquema, igual que lo haría FastAPI con Pydantic.

    esquema_json: JSON con un campo por clave, por ejemplo
      {"name": {"tipo": "str", "obligatorio": true},
       "price": {"tipo": "float", "obligatorio": true},
       "tax": {"tipo": "float", "obligatorio": false}}
      Tipos permitidos: str, int, float, bool.
    datos_json: el objeto JSON a validar, por ejemplo {"name": "Foo", "price": 45.2}

    Devuelve VÁLIDO o INVÁLIDO con el detalle de cada error.
    Validar un solo objeto por llamada."""
    try:
        esquema = json.loads(esquema_json)
        datos = json.loads(datos_json)
    except json.JSONDecodeError as e:
        return f"ERROR: JSON mal formado ({e})."

    campos = {}
    for nombre, spec in esquema.items():
        tipo = TIPOS.get(spec.get("tipo"))
        if tipo is None:
            return f"ERROR: tipo '{spec.get('tipo')}' no permitido en el campo '{nombre}'. Usar: {list(TIPOS)}."
        # Obligatorio → sin valor por defecto. Opcional → acepta None, como en FastAPI.
        campos[nombre] = (tipo, ...) if spec.get("obligatorio") else (tipo | None, None)

    Modelo = create_model("ModeloValidado", **campos)
    try:
        Modelo.model_validate(datos)
        return f"VÁLIDO: {datos_json} cumple el esquema."
    except ValidationError as e:
        errores = "; ".join(f"campo '{'.'.join(map(str, err['loc']))}': {err['msg']}" for err in e.errors())
        return f"INVÁLIDO: {datos_json} → {errores}"


PROMPT_ANALISTA = """Sos un agente de análisis especializado en validación de esquemas.
Tu única función es validar datos JSON usando tu herramienta, a partir del esquema
que te den.

Reglas:
- NO busques información nueva: trabajá solo con los datos recibidos.
- Validá SIEMPRE con la herramienta (un JSON por llamada); no decidas a ojo.
- Respondé en español, de forma breve: para cada JSON indicá si es válido y, si no, por qué."""


def crear_nodo_analista(llm: BaseChatModel):
    # Herramientas acotadas: este agente SOLO puede validar.
    agente = create_agent(model=llm, tools=[validar_json_contra_esquema], system_prompt=PROMPT_ANALISTA)

    async def nodo_analista(state: AgentState) -> dict:
        instruccion = state["instruccion"]

        # Contexto necesario y nada más: la instrucción + el último aporte del
        # investigador (de donde sale el esquema). No recibe el resto del historial.
        investigaciones = [c for c in state.get("contribuciones", []) if c["agente"] == "investigador"]
        contexto = investigaciones[-1]["aporte"] if investigaciones else "(sin datos del investigador)"
        mensaje = f"{instruccion}\n\nDatos del investigador:\n{contexto}"

        resultado = await agente.ainvoke({"messages": [HumanMessage(content=mensaje)]})
        respuesta = texto(resultado["messages"][-1])
        paso = state.get("pasos", 0) + 1

        return {
            "messages": [AIMessage(content=respuesta, name="analista")],
            "contribuciones": [
                {"agente": "analista", "instruccion": instruccion, "aporte": respuesta, "paso": paso}
            ],
            "pasos": paso,
        }

    return nodo_analista
