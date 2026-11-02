"""Prueba de carga: 5 peticiones concurrentes contra la API.

Uso (con la API levantada):
    python scripts/prueba_carga.py              # deja los jobs esperando aprobación
    python scripts/prueba_carga.py --aprobar    # además los aprueba y espera a que terminen
"""

import argparse
import asyncio
import time

import httpx

API = "http://localhost:8000"

CONSULTAS = [
    'Según la documentación de FastAPI, ¿qué campos tiene el modelo Item del tutorial de Request Body y cuáles '
    'son obligatorios? Validá si {"name": "Teclado", "price": 45.2} y {"description": "Sin nombre", "price": "barato"} serían aceptados.',
    'Según la documentación, ¿de qué tipo es item_id en el tutorial de Path Parameters? Validá si {"item_id": 3} '
    'y {"item_id": "foo"} cumplen ese esquema.',
    'Según la documentación de Query Parameters, ¿qué parámetros tienen valor por defecto y cuál es opcional? '
    'Validá {"skip": 0, "limit": 10} y {"skip": "cero", "limit": 10} contra ese esquema.',
    'Según el tutorial de Request Body, ¿FastAPI convierte tipos automáticamente? Validá si {"name": "Mouse", '
    '"price": "45"} sería aceptado con el modelo Item.',
    'Con el modelo Item del tutorial de Request Body, validá si {"name": "Monitor", "price": 120.5, "tax": "alto"} '
    'es válido y explicá qué dice la documentación sobre los campos opcionales.',
]

FINALES = {"DONE", "FAILED"}


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str], timeout: float = 300) -> dict:
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < timeout:
        job = (await cliente.get(f"{API}/tasks/{job_id}")).json()
        if job["status"] in estados:
            return job
        await asyncio.sleep(1)
    return {"job_id": job_id, "status": "TIMEOUT"}


async def main(aprobar: bool) -> None:
    async with httpx.AsyncClient(timeout=30) as cliente:
        inicio = time.perf_counter()
        respuestas = await asyncio.gather(*[cliente.post(f"{API}/tasks", json={"query": q}) for q in CONSULTAS])
        print(f"📨 5 POST concurrentes respondidos en {time.perf_counter() - inicio:.2f}s")
        ids = [r.json()["job_id"] for r in respuestas]

        jobs = await asyncio.gather(*[esperar(cliente, i, {"WAITING_APPROVAL"} | FINALES) for i in ids])
        print(f"⏸️  Primera fase terminada en {time.perf_counter() - inicio:.1f}s")
        for j in jobs:
            print(f"   {j['job_id'][:8]}  {j['status']}  {j.get('error', '')}")

        if not aprobar:
            print("\nAprobá con: POST /tasks/{job_id}/approve  (o corré de nuevo con --aprobar)")
            return

        pendientes = [j["job_id"] for j in jobs if j["status"] == "WAITING_APPROVAL"]
        await asyncio.gather(*[
            cliente.post(f"{API}/tasks/{i}/approve", json={"aprobado": True, "comentario": "Aprobado en prueba de carga"})
            for i in pendientes
        ])
        finales = await asyncio.gather(*[esperar(cliente, i, FINALES) for i in pendientes])
        print(f"\n✅ Terminado en {time.perf_counter() - inicio:.1f}s")
        for j in finales:
            print(f"   {j['job_id'][:8]}  {j['status']}  informe: {j.get('informe_path')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aprobar", action="store_true", help="aprobar automáticamente los jobs pausados")
    asyncio.run(main(parser.parse_args().aprobar))
