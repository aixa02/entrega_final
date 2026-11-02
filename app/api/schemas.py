"""Contratos de datos de la API (Pydantic).

Las entradas se validan antes de llegar al código (un body inválido devuelve 422)
y las salidas se declaran con response_model: FastAPI las valida, filtra campos
de más y las documenta en /docs.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.jobs import JobStatus

# ---------- Entradas ----------


class TaskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=2000, description="Consulta para el orquestador")


class AprobacionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    aprobado: bool = Field(description="true guarda el informe; false lo descarta")
    comentario: str | None = Field(None, max_length=500)


# ---------- Salidas ----------


class TaskCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus


class AprobacionResponse(BaseModel):
    job_id: str
    status: JobStatus
    aprobado: bool


class MensajeConversacion(BaseModel):
    autor: str = Field(description="usuario, supervisor, investigador, analista o sintesis")
    contenido: str


class ContribucionDetalle(BaseModel):
    paso: int
    agente: str
    instruccion: str
    aporte: str


class ConversacionResponse(BaseModel):
    """Estado del grafo leído directamente del checkpointer (Redis)."""

    job_id: str
    mensajes: list[MensajeConversacion]
    contribuciones: list[ContribucionDetalle]
    siguiente_nodo: list[str] = Field(description="Nodo pendiente; vacío si el grafo terminó")
    checkpoints: int = Field(description="Cantidad de estados guardados para esta conversación")


class HealthResponse(BaseModel):
    status: Literal["ok"]
