"""
Serviço de inferência — consome métricas do gRPC e classifica em tempo real.
Expõe estado via HTTP para o api-gateway consultar.

Fluxo: grpc-ingestion → [stream gRPC] → inference worker → resultado em memória → api-gateway
"""
import time
import os
import sys
import threading
from fastapi import FastAPI
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "1.0"))
HTTP_PORT = int(os.getenv("ML_HTTP_PORT", "8001"))

# Estado compartilhado (thread-safe com lock)
_lock = threading.Lock()
_results: list = []
MAX_HISTORY = 200


def _add_result(result: dict):
    with _lock:
        _results.append(result)
        if len(_results) > MAX_HISTORY:
            _results.pop(0)


def get_results() -> list:
    with _lock:
        return list(_results)


# ── FastAPI interna (consumida pelo api-gateway via HTTP) ──────────────────────
app = FastAPI(title="ML Inference Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "results_buffered": len(_results)}


@app.get("/results")
def results(limit: int = 50):
    data = get_results()
    return {"count": len(data), "results": data[-limit:]}


@app.post("/predict")
def predict_single(metric: dict):
    """Endpoint para inferência sob demanda (usado pelo api-gateway)."""
    try:
        from model import load_model, predict
        clf, scaler = load_model()
        return predict(metric, clf, scaler)
    except Exception as e:
        return {"error": str(e)}


# ── Worker de inferência (consome gRPC stream) ─────────────────────────────────
def _start_inference_worker():
    """Carrega modelo e inicia consumo do stream gRPC em background."""
    try:
        from model import load_model, predict
        clf, scaler = load_model()
        print("[Inference] Modelo carregado.")
    except FileNotFoundError:
        print("[Inference] Modelo não encontrado. Execute trainer.py primeiro.")
        clf, scaler = None, None

    def on_metric(metric: dict):
        if clf is None:
            return
        try:
            result = predict(metric, clf, scaler)
            _add_result(result)
            if result["status"] != "NORMAL":
                print(f"[ALERTA] {result['node_id']}: {result['status']} → {result['action']}")
        except Exception as e:
            print(f"[Inference] Erro ao classificar: {e}")

    from grpc_client import stream_from_grpc
    t = threading.Thread(target=stream_from_grpc, args=(on_metric,), daemon=True)
    t.start()
    print("[Inference] Worker gRPC iniciado.")


@app.on_event("startup")
def startup():
    _start_inference_worker()


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=HTTP_PORT, log_level="info")
