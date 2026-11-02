"""Crea el índice Serverless de Pinecone si no existe (Pre-entrega 4).

Uso:  python -m scripts.setup_index
"""

import asyncio

from pinecone import PineconeAsyncio, ServerlessSpec

from app.core.config import get_settings


async def setup_index() -> None:
    s = get_settings()
    async with PineconeAsyncio(api_key=s.pinecone_api_key) as pc:
        if await pc.has_index(s.pinecone_index):
            print(f"♻️ El índice '{s.pinecone_index}' ya existe")
        else:
            print(f"🆕 Creando índice '{s.pinecone_index}' (dimensión {s.embedding_dim})...")
            await pc.create_index(
                name=s.pinecone_index,
                dimension=s.embedding_dim,  # tiene que coincidir con el modelo de embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
            while not (await pc.describe_index(s.pinecone_index)).status["ready"]:
                await asyncio.sleep(1)
            print("✅ Índice listo")

        host = (await pc.describe_index(s.pinecone_index)).host
        async with pc.IndexAsyncio(host=host) as idx:
            print(await idx.describe_index_stats())


if __name__ == "__main__":
    asyncio.run(setup_index())
