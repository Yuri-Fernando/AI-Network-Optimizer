"""
Simulador de métricas de rede (imita tráfego de gNB no Open RAN).
Gera dados normais e congestionados aleatoriamente.
"""
import random
import time


NODES = ["gNB-001", "gNB-002", "gNB-003"]


def _normal_metric(node_id: str) -> dict:
    return {
        "latency": round(random.uniform(5, 30), 2),
        "throughput": round(random.uniform(80, 150), 2),
        "packet_loss": round(random.uniform(0.0, 1.0), 3),
        "jitter": round(random.uniform(1, 5), 2),
        "timestamp": int(time.time() * 1000),
        "node_id": node_id,
    }


def _congested_metric(node_id: str) -> dict:
    return {
        "latency": round(random.uniform(80, 250), 2),
        "throughput": round(random.uniform(5, 40), 2),
        "packet_loss": round(random.uniform(5.0, 20.0), 3),
        "jitter": round(random.uniform(20, 60), 2),
        "timestamp": int(time.time() * 1000),
        "node_id": node_id,
    }


def generate_metric() -> dict:
    """Gera uma métrica aleatória para um nó aleatório (30% chance de congestionamento)."""
    node = random.choice(NODES)
    if random.random() < 0.3:
        return _congested_metric(node)
    return _normal_metric(node)


def stream_metrics(interval: float = 1.0):
    """Gerador infinito de métricas."""
    while True:
        yield generate_metric()
        time.sleep(interval)
