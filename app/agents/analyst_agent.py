"""Agente de análisis: valida datos JSON contra un esquema (validación de esquemas)."""

import json
from typing import Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, create_model

from app.core.llm import texto
from app.graph.state import AgentState

# Tipos permitidos en el esquema. Se mapean a mano (sin eval) para que el LLM
# no pueda inyectar código: el Literal de CampoEsquema solo admite estos nombres.
TIPOS = {"str": str, "int": int, "float": float, "bool": bool}


class CampoEsquema(BaseModel):
    nombre: str = Field(min_length=1, description="Nombre del campo, por ejemplo 'price'")
    tipo: Literal["str", "int", "float", "bool"]
    obligatorio: bool


class ValidacionInput(BaseModel):
    campos: list[CampoEsquema] = Field(min_length=1, description="Esquema: un elemento por campo")
    datos_json: str = Field(description='Objeto JSON a validar, por ejemplo {"name": "Foo", "price": 45.2}')


class ErrorCampo(BaseModel):
    campo: str
    mensaje: str


class ResultadoValidacion(BaseModel):
    valido: bool
    datos: str
    errores: list[ErrorCampo]


@tool(args_schema=ValidacionInput)
async def validar_json_contra_esquema(campos: list[CampoEsquema], datos_json: str) -> str:
    """Valida si un objeto JSON cumple un esquema, igual que lo haría FastAPI con Pydantic.
    Validar un solo objeto JSON por llamada. Devuelve un JSON con 'valido' y la lista de errores."""
    try:
        datos = json.loads(datos_json)
        if not isinstance(datos, dict):
            raise ValueError("se esperaba un objeto JSON")
    except ValueError as e:
        error = ErrorCampo(campo="(JSON)", mensaje=f"JSON mal formado: {e}")
        return ResultadoValidacion(valido=False, datos=datos_json, errores=[error]).model_dump_json()

    # Obligatorio → sin valor por defecto. Opcional → acepta None, como en FastAPI.
    definicion = {
        c.nombre: (TIPOS[c.tipo], ...) if c.obligatorio else (TIPOS[c.tipo] | None, None) for c in campos
    }
    Modelo = create_model("ModeloValidado", **definicion)
    try:
        Modelo.model_validate(datos)
        errores: list[ErrorCampo] = []
    except ValidationError as e:
        errores = [ErrorCampo(campo=".".join(map(str, err["loc"])), mensaje=err["msg"]) for err in e.errors()]
    return ResultadoValidacion(valido=not errores, datos=datos_json, errores=errores).model_dump_json()


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
