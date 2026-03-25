"""
API Gateway — ponto de entrada REST do sistema AI Network Optimizer.
Inspirado na interface norte do RIC (A1/O1 interfaces do O-RAN).
"""
import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(
    title="AI Network Optimizer",
    description=(
        "Sistema de detecção de anomalias em métricas de rede inspirado em xApps do O-RAN. "
        "Ingere dados via gRPC, classifica via ML e expõe decisões via REST."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "AI Network Optimizer — API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "status_endpoint": "/network/status",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Inicia inference worker em background quando rodar standalone
def _start_inference_worker():
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml-service/app"))
        from inference import run_inference_loop
        t = threading.Thread(target=run_inference_loop, daemon=True)
        t.start()
        print("[API Gateway] Worker de inferência iniciado em background.")
    except Exception as e:
        print(f"[API Gateway] Worker de inferência não disponível: {e}")


@app.on_event("startup")
def startup_event():
    _start_inference_worker()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
