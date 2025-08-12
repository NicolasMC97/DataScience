# load_test.py
import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Generador de coordenadas Bogotá ---
def gen_coord_bogota() -> Tuple[float, float]:
    lat = round(random.uniform(4.48, 4.83), 6)
    lon = round(random.uniform(-74.20, -73.99), 6)
    return lat, lon

@dataclass
class Sample:
    ok: bool
    status: Optional[int]
    latency_ms: float
    error: Optional[str]

# --- Cliente con requests (threaded) ---
def run_requests(url: str, total: int, concurrency: int, timeout: float) -> List[Sample]:
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    samples: List[Sample] = []

    def do_one() -> Sample:
        lat, lon = gen_coord_bogota()
        payload = {"latitude": lat, "longitude": lon}
        t0 = time.perf_counter()
        try:
            r = session.post(url, json=payload, timeout=timeout)
            # Opcional: validar JSON sin romper
            try:
                _ = r.json()
            except Exception:
                pass
            lat_ms = (time.perf_counter() - t0) * 1000
            return Sample(ok=r.ok, status=r.status_code, latency_ms=lat_ms, error=None if r.ok else f"HTTP {r.status_code}")
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000
            return Sample(ok=False, status=None, latency_ms=lat_ms, error=str(e))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(do_one) for _ in range(total)]
        for f in as_completed(futs):
            samples.append(f.result())
    session.close()
    return samples

# --- Cliente con aiohttp (async) ---
async def run_aiohttp(url: str, total: int, concurrency: int, timeout: float) -> List[Sample]:
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    samples: List[Sample] = []

    async def do_one(session: aiohttp.ClientSession):
        lat, lon = gen_coord_bogota()
        payload = {"latitude": lat, "longitude": lon}
        t0 = time.perf_counter()
        async with sem:
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    # Opcional: validar JSON sin romper
                    try:
                        _ = await resp.json()
                    except Exception:
                        pass
                    lat_ms = (time.perf_counter() - t0) * 1000
                    samples.append(Sample(ok=200 <= resp.status < 300, status=resp.status, latency_ms=lat_ms,
                                          error=None if 200 <= resp.status < 300 else f"HTTP {resp.status}"))
            except Exception as e:
                lat_ms = (time.perf_counter() - t0) * 1000
                samples.append(Sample(ok=False, status=None, latency_ms=lat_ms, error=str(e)))

    conn = aiohttp.TCPConnector(limit=0)  # sin límite por conexión
    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}, connector=conn) as session:
        tasks = [asyncio.create_task(do_one(session)) for _ in range(total)]
        await asyncio.gather(*tasks)
    return samples

# --- Métricas ---
def percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1

def summarize(samples: List[Sample], started_at: float, ended_at: float):
    total = len(samples)
    ok = sum(1 for s in samples if s.ok)
    fail = total - ok
    dur_s = ended_at - started_at
    rps = total / dur_s if dur_s > 0 else float("inf")
    latencies = sorted(s.latency_ms for s in samples)
    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    mean = statistics.fmean(latencies) if latencies else float("nan")
    print("\n=== Resultados ===")
    print(f"Total req: {total} | OK: {ok} | Fail: {fail}")
    print(f"Duración: {dur_s:.2f}s | RPS: {rps:.2f}")
    print(f"Latencia (ms): mean {mean:.1f} | p50 {p50:.1f} | p90 {p90:.1f} | p95 {p95:.1f} | p99 {p99:.1f}")
    if fail:
        errs = {}
        for s in samples:
            if not s.ok:
                key = s.error or f"HTTP {s.status}"
                errs[key] = errs.get(key, 0) + 1
        print("Errores más comunes:")
        for k, v in sorted(errs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {k}: {v}")

def main():
    ap = argparse.ArgumentParser(description="Cargar API /predict con coordenadas de Bogotá en paralelo.")
    ap.add_argument("--url", default="http://localhost:9000/predict", help="URL del endpoint")
    ap.add_argument("-n", "--requests", type=int, default=200, help="Número total de requests")
    ap.add_argument("-c", "--concurrency", type=int, default=500, help="Concurrencia (hilos/corrutinas)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Timeout por request (segundos)")
    ap.add_argument("--engine", choices=["requests", "aiohttp"], default="requests", help="Motor HTTP")
    ap.add_argument("--save", help="Guardar resultados crudos en JSON (ruta)")
    args = ap.parse_args()

    random.seed(42)  # reproducible

    t0 = time.perf_counter()
    if args.engine == "requests":
        samples = run_requests(args.url, args.requests, args.concurrency, args.timeout)
    else:
        samples = asyncio.run(run_aiohttp(args.url, args.requests, args.concurrency, args.timeout))
    t1 = time.perf_counter()

    summarize(samples, t0, t1)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump([s.__dict__ for s in samples], f, ensure_ascii=False, indent=2)
        print(f"Guardado: {args.save}")

if __name__ == "__main__":
    main()
