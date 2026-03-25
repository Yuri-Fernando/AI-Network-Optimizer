"""
Modelo de detecção de anomalias de rede.
Classifica métricas como: NORMAL | CONGESTIONADO | DEGRADADO
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/scaler.pkl")

LABELS = {0: "NORMAL", 1: "CONGESTIONADO", 2: "DEGRADADO"}


def extract_features(metric: dict) -> np.ndarray:
    """Extrai vetor de features de uma métrica de rede."""
    return np.array([
        metric["latency"],
        metric["throughput"],
        metric["packet_loss"],
        metric["jitter"],
    ]).reshape(1, -1)


def load_model():
    """Carrega modelo e scaler do disco."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute trainer.py primeiro.")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def predict(metric: dict, clf, scaler) -> dict:
    """Classifica uma métrica e retorna status + ação recomendada."""
    features = extract_features(metric)
    features_scaled = scaler.transform(features)
    label_id = clf.predict(features_scaled)[0]
    proba = clf.predict_proba(features_scaled)[0]

    status = LABELS[label_id]
    confidence = round(float(max(proba)), 3)
    action = _recommend_action(status, metric)

    return {
        "node_id": metric.get("node_id", "unknown"),
        "status": status,
        "confidence": confidence,
        "action": action,
        "raw_metrics": {
            "latency": metric["latency"],
            "throughput": metric["throughput"],
            "packet_loss": metric["packet_loss"],
            "jitter": metric["jitter"],
        },
    }


def _recommend_action(status: str, metric: dict) -> str:
    if status == "NORMAL":
        return "NENHUMA"
    if status == "CONGESTIONADO":
        return "BALANCEAR_CARGA"
    if status == "DEGRADADO":
        if metric["packet_loss"] > 10:
            return "REROUTING_EMERGENCIAL"
        return "REDUZIR_THROUGHPUT"
    return "MONITORAR"
