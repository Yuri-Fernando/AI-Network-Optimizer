"""
Rotas REST do API Gateway.
Consulta o ml-service via HTTP — fluxo real entre serviços.
Equivalente à interface A1/O1 do O-RAN (norte do RIC).
"""
import os
import httpx
from fastapi import APIRouter, HTTPException

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")

router = APIRouter(prefix="/network", tags=["network"])


def _call_ml(path: str, params: dict = None) -> dict:
    """Chama ml-service via HTTP com timeout e tratamento de erro."""
    try:
        url = f"{ML_SERVICE_URL}{path}"
        r = httpx.get(url, params=params, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"ML service indisponível em {ML_SERVICE_URL}. Verifique se o container está rodando."
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ML service timeout.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def get_network_status():
    """
    Retorna status atual de todos os nós.
    Consulta ml-service via HTTP → agrega por nó → determina status geral.
    """
    data = _call_ml("/results", {"limit": 200})
    results = data.get("results", [])

    if not results:
        return {
            "overall_status": "SEM_DADOS",
            "message": "ML service ainda não processou métricas.",
            "nodes": [],
        }

    # Última entrada por nó
    node_map: dict = {}
    for r in reversed(results):
        nid = r.get("node_id", "unknown")
        if nid not in node_map:
            node_map[nid] = r

    overall = "NORMAL"
    for r in node_map.values():
        if r["status"] == "DEGRADADO":
            overall = "DEGRADADO"
            break
        if r["status"] == "CONGESTIONADO":
            overall = "CONGESTIONADO"

    return {
        "overall_status": overall,
        "nodes": list(node_map.values()),
        "total_nodes": len(node_map),
    }


@router.get("/history")
def get_history(limit: int = 20):
    """Histórico das últimas inferências do ml-service."""
    data = _call_ml("/results", {"limit": limit})
    return {
        "count": data.get("count", 0),
        "history": data.get("results", []),
    }


@router.get("/alerts")
def get_alerts():
    """Retorna somente anomalias detectadas."""
    data = _call_ml("/results", {"limit": 200})
    results = data.get("results", [])
    alerts = [r for r in results if r.get("status") != "NORMAL"]
    return {
        "alert_count": len(alerts),
        "alerts": alerts[-50:],
    }


@router.get("/node/{node_id}")
def get_node_status(node_id: str):
    """Status do último ciclo de um nó específico."""
    data = _call_ml("/results", {"limit": 200})
    results = data.get("results", [])
    node_results = [r for r in results if r.get("node_id") == node_id]
    if not node_results:
        raise HTTPException(status_code=404, detail=f"Nó '{node_id}' não encontrado.")
    return node_results[-1]


@router.post("/predict")
def predict_on_demand(metric: dict):
    """
    Inferência sob demanda — envia métrica ao ml-service e retorna classificação.
    Útil para testes pontuais e integração com sistemas externos.
    """
    try:
        r = httpx.post(f"{ML_SERVICE_URL}/predict", json=metric, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="ML service indisponível.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
