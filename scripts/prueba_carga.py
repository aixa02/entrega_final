"""Prueba de punta a punta: 5 consultas concurrentes contra la API.

Cada consulta sigue su propio recorrido: se encola, espera la aprobación humana,
se aprueba (con --aprobar) y espera a terminar. Las 5 avanzan en paralelo, así que
una consulta lenta no frena la aprobación de las demás.

Uso (con la API levantada):
    python scripts/prueba_carga.py              # deja los jobs esperando aprobación
    python scripts/prueba_carga.py --aprobar    # además los aprueba y espera a que terminen
"""

import argparse
import asyncio
import time

import httpx

API = "http://localhost:8000"
TIMEOUT_SEGUNDOS = 15 * 60  # tope por consulta: el límite de pedidos del LLM hace lento el proceso

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


async def esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str], limite: float) -> dict:
    while time.perf_counter() < limite:
        job = (await cliente.get(f"{API}/tasks/{job_id}")).json()
        if job["status"] in estados:
            return job
        await asyncio.sleep(2)
    return {"job_id": job_id, "status": "TIMEOUT"}


async def una_prueba(cliente: httpx.AsyncClient, n: int, consulta: str, aprobar: bool) -> dict:
    inicio = time.perf_counter()
    limite = inicio + TIMEOUT_SEGUNDOS
    job_id = (await cliente.post(f"{API}/tasks", json={"query": consulta})).json()["job_id"]
    print(f"📨 [{n}] encolada → {job_id}")

    job = await esperar(cliente, job_id, {"WAITING_APPROVAL"} | FINALES, limite)
    print(f"⏸️  [{n}] {job['status']} en {time.perf_counter() - inicio:.0f}s")

    if aprobar and job["status"] == "WAITING_APPROVAL":
        await cliente.post(f"{API}/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "Aprobado en la prueba"})
        job = await esperar(cliente, job_id, FINALES, limite)
        print(f"✅ [{n}] {job['status']} en {time.perf_counter() - inicio:.0f}s")

    return {**job, "job_id": job_id, "segundos": round(time.perf_counter() - inicio)}


async def main(aprobar: bool) -> None:
    inicio = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as cliente:
        resultados = await asyncio.gather(
            *[una_prueba(cliente, n, q, aprobar) for n, q in enumerate(CONSULTAS, 1)]
        )

    print(f"\n{'=' * 70}\nResumen ({time.perf_counter() - inicio:.0f}s en total)\n{'=' * 70}")
    for n, r in enumerate(resultados, 1):
        detalle = r.get("error") or r.get("informe_path") or ""
        print(f"[{n}] {r['job_id']}  {r['status']:<16} {r['segundos']:>4}s  {detalle}")

    pendientes = [r["job_id"] for r in resultados if r["status"] == "WAITING_APPROVAL"]
    if pendientes:
        print("\nPendientes de aprobación: POST /tasks/{job_id}/approve")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aprobar", action="store_true", help="aprobar automáticamente cada job pausado")
    asyncio.run(main(parser.parse_args().aprobar))
