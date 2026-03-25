"""
Serviço de inferência — consome métricas do gRPC e expõe resultados via Redis/fila.
Roda como worker contínuo.
"""
import time
import json
import os
import sys

# Adiciona path do simulador (em dev, pode consumir localmente)
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "1.0"))

# Estado compartilhado em memória (simplificado; em prod use Redis)
_latest_results: list = []
MAX_HISTORY = 100


def get_latest_results() -> list:
    return list(_latest_results)


def add_result(result: dict):
    _latest_results.append(result)
    if len(_latest_results) > MAX_HISTORY:
        _latest_results.pop(0)


def run_inference_loop():
    """Loop de inferência: puxa métricas do simulador local e classifica."""
    # Import local do simulador (dentro do container, está no mesmo diretório)
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        from simulator import generate_metric
        from model import load_model, predict
    except ImportError as e:
        print(f"[Inference] Erro de importação: {e}")
        print("[Inference] Gerando dados mock para demonstração.")
        generate_metric = None
        load_model = None

    print("[Inference] Tentando carregar modelo...")
    clf, scaler = None, None
    if load_model:
        try:
            clf, scaler = load_model()
            print("[Inference] Modelo carregado com sucesso.")
        except FileNotFoundError:
            print("[Inference] Modelo não encontrado. Execute trainer.py primeiro.")

    print(f"[Inference] Worker iniciado. Intervalo: {INFERENCE_INTERVAL}s")

    while True:
        try:
            if generate_metric and clf:
                metric = generate_metric()
                result = predict(metric, clf, scaler)
                add_result(result)

                if result["status"] != "NORMAL":
                    print(f"[ALERTA] {result['node_id']}: {result['status']} -> {result['action']}")
            else:
                # Mock sem modelo carregado
                mock = {
                    "node_id": "gNB-001",
                    "status": "NORMAL",
                    "confidence": 0.95,
                    "action": "NENHUMA",
                    "raw_metrics": {"latency": 12.0, "throughput": 120.0, "packet_loss": 0.1, "jitter": 2.0},
                }
                add_result(mock)

        except Exception as e:
            print(f"[Inference] Erro: {e}")

        time.sleep(INFERENCE_INTERVAL)


if __name__ == "__main__":
    run_inference_loop()
