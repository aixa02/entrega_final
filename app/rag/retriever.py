"""Recuperador híbrido (BM25 + Pinecone) de la Pre-entrega 4."""

import json
import re
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


def tokenizar(texto: str) -> list[str]:
    """Pasa a minúsculas y separa por palabras, ignorando signos de puntuación."""
    return re.findall(r"\w+", texto.lower())


def cargar_chunks() -> list[Document]:
    with open(get_settings().chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]


@lru_cache(maxsize=1)
def crear_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().embedding_model)


class RAGSystem:
    """Recuperador híbrido: BM25 (léxico, en memoria) + Pinecone (semántico)."""

    def __init__(self, pesos: tuple[float, float] = (0.5, 0.5)):
        settings = get_settings()
        self.k = settings.rag_top_k

        self.bm25 = BM25Retriever.from_documents(cargar_chunks(), preprocess_func=tokenizar)
        self.bm25.k = self.k

        vectorstore = PineconeVectorStore(
            index_name=settings.pinecone_index,
            embedding=crear_embeddings(),
            namespace=settings.pinecone_namespace,
            pinecone_api_key=settings.pinecone_api_key,
        )
        self.vectorial = vectorstore.as_retriever(search_kwargs={"k": self.k})

        self.hibrido = EnsembleRetriever(retrievers=[self.bm25, self.vectorial], weights=list(pesos))

    async def abuscar(self, query: str, modo: str = "hibrido") -> list[Document]:
        """Devuelve los top-k documentos. modo: 'hibrido', 'bm25' o 'vectorial'.

        En modo async, el EnsembleRetriever ejecuta los dos buscadores en paralelo:
        - Pinecone: la consulta de red es async nativa (cliente PineconeAsyncio).
        - BM25 y el embedding de la pregunta son cálculo de CPU, así que LangChain
          los corre en un thread. No es I/O: no hay versión "async" posible, y en
          un thread no bloquean el event loop.
        """
        retriever = {"hibrido": self.hibrido, "bm25": self.bm25, "vectorial": self.vectorial}[modo]
        return (await retriever.ainvoke(query))[: self.k]


@lru_cache(maxsize=1)
def get_rag() -> RAGSystem:
    """Se crea una sola vez: cargar el modelo de embeddings y BM25 es lento."""
    return RAGSystem()
