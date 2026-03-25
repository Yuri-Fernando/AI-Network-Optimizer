"""
API Gateway — ponto de entrada REST do sistema AI Network Optimizer.
Delega toda inferência ao ml-service via HTTP (serviços desacoplados).
Equivalente à interface norte do RIC (A1/O1) no O-RAN.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(
    title="AI Network Optimizer — API Gateway",
    description=(
        "Interface REST do sistema de detecção de anomalias de rede. "
        "Consome decisões do ML Service e as expõe para clientes externos. "
        "Arquitetura: gRPC Ingestion → ML Service → API Gateway (este serviço)."
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
        "architecture": "grpc-ingestion → ml-service → api-gateway",
        "docs": "/docs",
        "endpoints": {
            "status":   "/network/status",
            "alerts":   "/network/alerts",
            "history":  "/network/history",
            "predict":  "/network/predict  [POST]",
            "node":     "/network/node/{id}",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
