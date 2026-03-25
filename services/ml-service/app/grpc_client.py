"""
Cliente gRPC — consome stream de métricas do grpc-ingestion service.
Esse é o canal equivalente à E2 interface do O-RAN (RAN → near-RT RIC).
"""
import sys
import os
import time

GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")

# Importa código gerado pelo protoc (disponível no container via grpc-ingestion)
try:
    sys.path.insert(0, "/grpc-proto")  # montado via volume no Docker/K8s
    import metrics_pb2
    import metrics_pb2_grpc
    import grpc
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False


def stream_from_grpc(on_metric_callback, max_errors: int = 5):
    """
    Conecta ao grpc-ingestion e consome o stream de métricas.
    Chama on_metric_callback(dict) para cada frame recebido.
    Reconecta automaticamente em caso de falha (até max_errors vezes seguidas).
    """
    if not _GRPC_AVAILABLE:
        print("[gRPC Client] grpc não disponível — usando fallback do simulador local.")
        _fallback_loop(on_metric_callback)
        return

    consecutive_errors = 0
    target = f"{GRPC_HOST}:{GRPC_PORT}"

    while True:
        try:
            print(f"[gRPC Client] Conectando a {target}...")
            with grpc.insecure_channel(target) as channel:
                stub = metrics_pb2_grpc.NetworkMetricsStub(channel)
                empty = metrics_pb2.Empty()

                print(f"[gRPC Client] Stream iniciado de {target}")
                consecutive_errors = 0

                for metric in stub.StreamMetrics(empty):
                    on_metric_callback({
                        "latency":      metric.latency,
                        "throughput":   metric.throughput,
                        "packet_loss":  metric.packet_loss,
                        "jitter":       metric.jitter,
                        "timestamp":    metric.timestamp,
                        "node_id":      metric.node_id,
                    })

        except grpc.RpcError as e:
            consecutive_errors += 1
            print(f"[gRPC Client] Erro RPC ({consecutive_errors}/{max_errors}): {e.code()} — {e.details()}")
            if consecutive_errors >= max_errors:
                print("[gRPC Client] Máximo de erros atingido. Usando fallback local.")
                _fallback_loop(on_metric_callback)
                return
            time.sleep(2 ** consecutive_errors)  # backoff exponencial

        except Exception as e:
            consecutive_errors += 1
            print(f"[gRPC Client] Erro inesperado: {e}")
            time.sleep(3)


def _fallback_loop(on_metric_callback):
    """Fallback: usa simulador local quando gRPC não está disponível."""
    import importlib.util, pathlib
    sim_path = pathlib.Path(__file__).parent.parent.parent / "grpc-ingestion" / "app" / "simulator.py"

    if sim_path.exists():
        spec = importlib.util.spec_from_file_location("simulator", sim_path)
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        generate = sim.generate_metric
    else:
        # Fallback puro sem import externo
        import random, time as t
        def generate():
            is_bad = random.random() < 0.3
            return {
                "latency":     random.uniform(80, 250) if is_bad else random.uniform(5, 30),
                "throughput":  random.uniform(5, 40)   if is_bad else random.uniform(80, 150),
                "packet_loss": random.uniform(5, 20)   if is_bad else random.uniform(0, 1),
                "jitter":      random.uniform(20, 60)  if is_bad else random.uniform(1, 5),
                "timestamp":   int(t.time() * 1000),
                "node_id":     random.choice(["gNB-001", "gNB-002", "gNB-003"]),
            }

    print("[gRPC Client] Fallback ativo — gerando métricas localmente.")
    while True:
        on_metric_callback(generate())
        time.sleep(float(os.getenv("INFERENCE_INTERVAL", "1.0")))
