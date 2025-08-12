import asyncio
import aiohttp
import json
import time
import random
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse
import sys

class BogotaAPILoadTester:
    def __init__(self, api_url: str = "http://localhost:8000/predict", results_file: str = "api_results.json"):
        self.api_url = api_url
        self.results_file = results_file
        self.results = []
        
    def generar_coordenadas_bogota(self) -> tuple:
        """
        Genera coordenadas aleatorias dentro de los límites de Bogotá
        """
        lat_min, lat_max = 4.48, 4.83
        lon_min, lon_max = -74.20, -73.99
        
        lat = round(random.uniform(lat_min, lat_max), 6)
        lon = round(random.uniform(lon_min, lon_max), 6)
        
        return lat, lon
    
    async def make_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
        """
        Realiza una petición HTTP y mide su latencia
        """
        # Generar coordenadas aleatorias
        latitude, longitude = self.generar_coordenadas_bogota()
        
        payload = {
            "latitude": latitude,
            "longitude": longitude
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        request_timestamp = datetime.now().isoformat()
        
        result = {
            "request_id": request_id,
            "url": self.api_url,
            "method": "POST",
            "timestamp": request_timestamp,
            "payload": payload,
            "headers": headers
        }
        
        try:
            async with session.post(
                url=self.api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # en milisegundos
                
                # Leer respuesta
                try:
                    response_data = await response.json()
                except Exception as e:
                    response_data = await response.text()
                
                result.update({
                    "status_code": response.status,
                    "latency_ms": round(latency, 2),
                    "response_size": len(str(response_data)),
                    "success": 200 <= response.status < 400,
                    "response_data": response_data,
                    "error": None
                })
                
                print(f"Request {request_id}: {response.status} - {latency:.2f}ms - Coords: ({latitude}, {longitude})")
                
        except asyncio.TimeoutError:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            result.update({
                "status_code": None,
                "latency_ms": round(latency, 2),
                "response_size": 0,
                "success": False,
                "response_data": None,
                "error": "Timeout"
            })
            print(f"Request {request_id}: TIMEOUT - {latency:.2f}ms")
            
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            result.update({
                "status_code": None,
                "latency_ms": round(latency, 2),
                "response_size": 0,
                "success": False,
                "response_data": None,
                "error": str(e)
            })
            print(f"Request {request_id}: ERROR - {latency:.2f}ms - {str(e)}")
        
        return result
    
    async def run_concurrent_requests(self, concurrent_users: int, total_requests: int) -> List[Dict]:
        """
        Ejecuta peticiones concurrentes y retorna los resultados
        """
        print(f"🚀 Iniciando test de carga para API de Bogotá")
        print(f"📊 Usuarios concurrentes: {concurrent_users}")
        print(f"🔢 Total de peticiones: {total_requests}")
        print(f"🌐 Endpoint: {self.api_url}")
        print("-" * 60)
        
        # Crear conector con límite de conexiones
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Crear tareas para todas las peticiones
            tasks = []
            for i in range(total_requests):
                task = self.make_request(session, i + 1)
                tasks.append(task)
            
            # Ejecutar todas las tareas con limite de concurrencia
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def bounded_request(task):
                async with semaphore:
                    return await task
            
            # Ejecutar todas las peticiones
            start_time = time.time()
            results = await asyncio.gather(*[bounded_request(task) for task in tasks])
            end_time = time.time()
            
            total_duration = end_time - start_time
            
            print("-" * 60)
            print(f"✅ Test completado en {total_duration:.2f} segundos")
            
            return results, total_duration
    
    def save_results_to_json(self, results: List[Dict], append: bool = True):
        """
        Guarda los resultados en un archivo JSON
        """
        if append:
            # Cargar resultados existentes si existen
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_data.extend(results)
                        results = existing_data
                    else:
                        # Si no es una lista, crear nueva estructura
                        results = existing_data.get('requests', []) + results
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        # Guardar todos los resultados
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados en: {self.results_file}")
    
    def analyze_results(self, results: List[Dict], total_duration: float):
        """
        Analiza y muestra estadísticas de los resultados
        """
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        latencies = [r['latency_ms'] for r in successful_requests]
        
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS DE RENDIMIENTO")
        print("="*60)
        
        # Estadísticas generales
        print(f"Total de peticiones: {len(results)}")
        print(f"Peticiones exitosas: {len(successful_requests)} ({len(successful_requests)/len(results)*100:.1f}%)")
        print(f"Peticiones fallidas: {len(failed_requests)} ({len(failed_requests)/len(results)*100:.1f}%)")
        print(f"Duración total del test: {total_duration:.2f} segundos")
        print(f"Throughput: {len(results)/total_duration:.2f} requests/segundo")
        
        # Estadísticas de latencia
        if latencies:
            print(f"\n🚀 LATENCIAS (ms):")
            print(f"  Promedio: {statistics.mean(latencies):.2f}")
            print(f"  Mediana: {statistics.median(latencies):.2f}")
            print(f"  Mínima: {min(latencies):.2f}")
            print(f"  Máxima: {max(latencies):.2f}")
            
            if len(latencies) > 1:
                print(f"  Desviación estándar: {statistics.stdev(latencies):.2f}")
            
            # Percentiles
            sorted_latencies = sorted(latencies)
            p95 = sorted_latencies[int(0.95 * len(sorted_latencies))]
            p99 = sorted_latencies[int(0.99 * len(sorted_latencies))]
            print(f"  Percentil 95: {p95:.2f}")
            print(f"  Percentil 99: {p99:.2f}")
        
        # Códigos de estado
        status_codes = {}
        for result in results:
            status = result['status_code'] or 'Error'
            status_codes[status] = status_codes.get(status, 0) + 1
        
        print(f"\n📊 CÓDIGOS DE RESPUESTA:")
        for status, count in sorted(status_codes.items()):
            print(f"  {status}: {count} ({count/len(results)*100:.1f}%)")
        
        # Errores más comunes
        if failed_requests:
            error_types = {}
            for result in failed_requests:
                error = result['error'] or f"HTTP {result['status_code']}"
                error_types[error] = error_types.get(error, 0) + 1
            
            print(f"\n❌ TIPOS DE ERROR:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count}")

async def main():
    parser = argparse.ArgumentParser(description='Load tester para API de predicción de Bogotá')
    parser.add_argument('--concurrent', '-c', type=int, default=100,
                      help='Número de usuarios concurrentes (default: 10)')
    parser.add_argument('--requests', '-r', type=int, default=100,
                      help='Total de peticiones a realizar (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='api_results.json',
                      help='Archivo de salida para los resultados (default: api_results.json)')
    parser.add_argument('--url', '-u', type=str, default='http://localhost:8000/predict',
                      help='URL del API (default: http://localhost:8000/predict)')
    parser.add_argument('--no-append', action='store_true',
                      help='No hacer append, sobrescribir archivo de resultados')
    
    args = parser.parse_args()
    
    # Validaciones
    if args.concurrent <= 0:
        print("❌ El número de usuarios concurrentes debe ser mayor a 0")
        return
    
    if args.requests <= 0:
        print("❌ El número total de peticiones debe ser mayor a 0")
        return
    
    # Crear instancia del tester
    tester = BogotaAPILoadTester(api_url=args.url, results_file=args.output)
    
    try:
        # Ejecutar las pruebas
        results, total_duration = await tester.run_concurrent_requests(
            concurrent_users=args.concurrent,
            total_requests=args.requests
        )
        
        # Guardar resultados
        tester.save_results_to_json(results, append=not args.no_append)
        
        # Analizar y mostrar estadísticas
        tester.analyze_results(results, total_duration)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error durante el test: {str(e)}")

if __name__ == "__main__":
    # Para ejecutar directamente con parámetros por defecto
    if len(sys.argv) == 1:
        print("🔧 Ejecutando con parámetros por defecto...")
        print("Tip: Usa --help para ver todas las opciones disponibles")
        print()
    
    asyncio.run(main())

# Ejemplo de uso directo en el código (sin argumentos de línea de comandos):
"""
async def run_simple_test():
    tester = BogotaAPILoadTester()
    results, duration = await tester.run_concurrent_requests(concurrent_users=5, total_requests=20)
    tester.save_results_to_json(results)
    tester.analyze_results(results, duration)

# Descomenta la siguiente línea para ejecutar un test simple
# asyncio.run(run_simple_test())
"""