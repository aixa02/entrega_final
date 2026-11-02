"""Contratos de datos de la API (Pydantic)."""

from pydantic import BaseModel


class TaskRequest(BaseModel):
    query: str


class AprobacionRequest(BaseModel):
    aprobado: bool
    comentario: str | None = None
