"""
Treinamento do modelo de classificação de anomalias de rede.
Gera dados sintéticos, treina RandomForest e salva modelo.

Execute:
    python trainer.py
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

RANDOM_STATE = 42
N_SAMPLES = 3000


def generate_training_data(n: int = N_SAMPLES):
    """
    Gera dataset sintético com 3 classes:
      0 = NORMAL
      1 = CONGESTIONADO
      2 = DEGRADADO
    Features: [latency, throughput, packet_loss, jitter]
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # NORMAL: baixa latência, alto throughput, pouca perda
    normal = np.column_stack([
        rng.uniform(5, 30, n // 3),    # latency ms
        rng.uniform(80, 150, n // 3),  # throughput Mbps
        rng.uniform(0, 1, n // 3),     # packet_loss %
        rng.uniform(1, 5, n // 3),     # jitter ms
    ])
    labels_normal = np.zeros(n // 3, dtype=int)

    # CONGESTIONADO: alta latência, baixo throughput
    congested = np.column_stack([
        rng.uniform(80, 250, n // 3),
        rng.uniform(5, 40, n // 3),
        rng.uniform(5, 20, n // 3),
        rng.uniform(20, 60, n // 3),
    ])
    labels_congested = np.ones(n // 3, dtype=int)

    # DEGRADADO: latência extrema, throughput colapsado
    degraded = np.column_stack([
        rng.uniform(200, 500, n // 3),
        rng.uniform(1, 15, n // 3),
        rng.uniform(15, 40, n // 3),
        rng.uniform(50, 150, n // 3),
    ])
    labels_degraded = np.full(n // 3, 2, dtype=int)

    X = np.vstack([normal, congested, degraded])
    y = np.concatenate([labels_normal, labels_congested, labels_degraded])
    return X, y


def train():
    print("[Trainer] Gerando dados sintéticos...")
    X, y = generate_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[Trainer] Treinando RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print("\n[Trainer] Relatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "CONGESTIONADO", "DEGRADADO"]))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Trainer] Modelo salvo em {MODEL_PATH}")
    print(f"[Trainer] Scaler salvo em {SCALER_PATH}")


if __name__ == "__main__":
    train()
