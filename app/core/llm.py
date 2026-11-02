"""Cliente del LLM (Gemini) y utilidades para leer sus respuestas."""

from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


def crear_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    # Rate limiter compartido por TODOS los nodos y workers (hay un solo LLM).
    # En vez de chocar contra el límite de la API (error 429), cada llamada
    # espera su turno. max_bucket_size=1: no se acumulan pedidos para ráfagas.
    limitador = InMemoryRateLimiter(
        requests_per_second=settings.llm_rpm / 60,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )
    # temperature=0: decisiones de ruteo estables, no creatividad.
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        rate_limiter=limitador,
    )


def texto(mensaje: BaseMessage) -> str:
    """Devuelve solo el texto de un mensaje.

    Gemini a veces devuelve el contenido como lista de bloques
    ([{'type': 'text', 'text': '...', 'extras': {...}}]). Nos quedamos solo con el texto.
    """
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido)
    return str(contenido)
