"""Ingesta de la documentación en Pinecone (Pre-entrega 4).

Divide los Markdown de data/docs en chunks, los guarda en data/chunks.json (para BM25)
y sube los embeddings a Pinecone con metadatos.

Uso:  python -m scripts.ingest
"""

import asyncio
import json
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.rag.retriever import crear_embeddings

CATEGORIAS = {
    "path-params.md": "parametros",
    "query-params.md": "parametros",
    "body.md": "request",
    "handling-errors.md": "errores",
    "cors.md": "seguridad",
    "middleware.md": "middleware",
    "background-tasks.md": "tareas",
    "dependencies.md": "dependencias",
}


def cargar_y_dividir() -> list[Document]:
    loader = DirectoryLoader(
        str(get_settings().docs_dir), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(loader.load())

    contador: dict[str, int] = {}
    for chunk in chunks:
        fuente = os.path.basename(chunk.metadata["source"])
        idx = contador.get(fuente, 0)
        contador[fuente] = idx + 1
        chunk.metadata = {"source": fuente, "category": CATEGORIAS.get(fuente, "general"), "chunk_index": idx}
    return chunks


def guardar_chunks(chunks: list[Document]) -> None:
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(get_settings().chunks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def subir_a_pinecone(chunks: list[Document]) -> None:
    s = get_settings()
    vectorstore = PineconeVectorStore(
        index_name=s.pinecone_index,
        embedding=crear_embeddings(),
        namespace=s.pinecone_namespace,
        pinecone_api_key=s.pinecone_api_key,
    )
    # IDs fijos: si se vuelve a ingerir, los vectores se sobrescriben en vez de duplicarse.
    ids = [f"{c.metadata['source']}-{c.metadata['chunk_index']}" for c in chunks]
    await vectorstore.aadd_documents(chunks, ids=ids)


async def main() -> None:
    # Leer y dividir archivos locales es CPU/disco: se hace en un thread.
    chunks = await asyncio.to_thread(cargar_y_dividir)
    print(f"✂️ {len(chunks)} chunks generados")
    await asyncio.to_thread(guardar_chunks, chunks)
    print(f"💾 Guardados en {get_settings().chunks_path}")
    await subir_a_pinecone(chunks)
    print(f"📦 Subidos a Pinecone (namespace '{get_settings().pinecone_namespace}')")


if __name__ == "__main__":
    asyncio.run(main())
