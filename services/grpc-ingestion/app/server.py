"""
Servidor gRPC — ingestion de métricas de rede.
Simula o papel de um data collector em um near-RT RIC (xApp input layer).
"""
import time
import grpc
from concurrent import futures
from simulator import generate_metric

# Importa código gerado pelo protoc
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Gerado via: python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/metrics.proto
try:
    import metrics_pb2
    import metrics_pb2_grpc
except ImportError:
    print("AVISO: arquivos protobuf não gerados. Execute generate_proto.sh primeiro.")
    metrics_pb2 = None
    metrics_pb2_grpc = None

GRPC_PORT = os.getenv("GRPC_PORT", "50051")


class NetworkMetricsServicer:
    def StreamMetrics(self, request, context):
        """Stream contínuo de métricas — equivalente a um xApp recebendo dados do E2 interface."""
        print(f"[gRPC] Cliente conectado: {context.peer()}")
        while context.is_active():
            raw = generate_metric()
            if metrics_pb2:
                yield metrics_pb2.Metric(**raw)
            else:
                # fallback sem proto gerado (modo demo)
                time.sleep(1)

    def GetSnapshot(self, request, context):
        """Retorna uma única métrica atual."""
        raw = generate_metric()
        if metrics_pb2:
            return metrics_pb2.Metric(**raw)
        return None


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    if metrics_pb2_grpc:
        metrics_pb2_grpc.add_NetworkMetricsServicer_to_server(
            NetworkMetricsServicer(), server
        )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"[gRPC] Servidor rodando na porta {GRPC_PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
