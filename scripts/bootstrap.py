"""Prepara la base de conocimiento antes de levantar la API.

Es idempotente: se puede correr en cada arranque.
- Si el índice de Pinecone no existe, lo crea.
- Si el namespace está vacío, ingiere los documentos de data/docs (y regenera
  data/chunks.json para BM25).
- Si ya hay vectores, no hace nada.

Así el proyecto arranca en una máquina limpia con solo completar el .env.

Uso:  python -m scripts.bootstrap
"""

import asyncio

from pinecone import PineconeAsyncio

from app.core.config import get_settings
from scripts import ingest
from scripts.setup_index import setup_index


async def vectores_en_namespace() -> int:
    s = get_settings()
    async with PineconeAsyncio(api_key=s.pinecone_api_key) as pc:
        host = (await pc.describe_index(s.pinecone_index)).host
        async with pc.IndexAsyncio(host=host) as idx:
            stats = await idx.describe_index_stats()
    resumen = (stats.namespaces or {}).get(s.pinecone_namespace)
    return resumen.vector_count if resumen else 0


async def bootstrap() -> None:
    s = get_settings()
    await setup_index()

    cantidad = await vectores_en_namespace()
    if cantidad and s.chunks_path.exists():
        print(f"✅ Base de conocimiento lista: {cantidad} vectores en '{s.pinecone_namespace}'")
        return

    print("📥 Base de conocimiento vacía o incompleta: ingiriendo data/docs...")
    await ingest.main()


if __name__ == "__main__":
    asyncio.run(bootstrap())
