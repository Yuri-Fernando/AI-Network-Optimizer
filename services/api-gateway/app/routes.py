"""
Rotas REST do API Gateway.
Expõe decisões do ml-service para consumo externo (dashboard, orquestrador, xApp client).
"""
import sys
import os
from fastapi import APIRouter, HTTPException
from typing import Optional

# Importa o estado do inference worker (quando rodando no mesmo processo)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml-service/app"))

router = APIRouter(prefix="/network", tags=["network"])

# Importação condicional do inference (integração interna)
try:
    from inference import get_latest_results
    _has_inference = True
except ImportError:
    _has_inference = False

    def get_latest_results():
        return []


@router.get("/status")
def get_network_status():
    """Retorna o status atual da rede (última inferência de todos os nós)."""
    results = get_latest_results()
    if not results:
        return {
            "status": "SEM_DADOS",
            "message": "Nenhuma métrica processada ainda. Aguarde o worker de inferência.",
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
    """Retorna histórico de inferências (últimas N entradas)."""
    results = get_latest_results()
    return {
        "count": len(results),
        "history": results[-limit:],
    }


@router.get("/node/{node_id}")
def get_node_status(node_id: str):
    """Retorna status de um nó específico."""
    results = get_latest_results()
    node_results = [r for r in results if r.get("node_id") == node_id]
    if not node_results:
        raise HTTPException(status_code=404, detail=f"Nó '{node_id}' não encontrado.")
    return node_results[-1]


@router.get("/alerts")
def get_alerts():
    """Retorna somente entradas com anomalias detectadas."""
    results = get_latest_results()
    alerts = [r for r in results if r.get("status") != "NORMAL"]
    return {
        "alert_count": len(alerts),
        "alerts": alerts[-50:],
    }
