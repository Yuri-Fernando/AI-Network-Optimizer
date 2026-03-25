"""
Schema compartilhado de métricas — contrato entre serviços.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class NetworkMetric:
    latency: float       # ms
    throughput: float    # Mbps
    packet_loss: float   # %
    jitter: float        # ms
    timestamp: int       # unix epoch ms
    node_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkMetric":
        return cls(**d)


@dataclass
class InferenceResult:
    node_id: str
    status: str          # NORMAL | CONGESTIONADO | DEGRADADO
    confidence: float
    action: str
    raw_metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)
