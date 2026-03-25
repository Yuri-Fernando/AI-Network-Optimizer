# AI Network Optimizer

> Sistema de detecção de anomalias em métricas de rede, inspirado na arquitetura **xApp / near-RT RIC do ecossistema O-RAN (Open RAN)**.

---

## Visão Geral

Redes de telecomunicações modernas (4G/5G) geram volumes massivos de métricas em tempo real — latência, throughput, perda de pacotes, jitter. Detectar degradações antes que impactem usuários é um dos maiores desafios operacionais do setor.

Este projeto simula exatamente o que acontece dentro de um **RIC (RAN Intelligent Controller)** do O-RAN:

- Um **xApp** coleta métricas da rede via stream contínuo
- Um modelo de IA classifica o estado da rede em tempo quase real
- Decisões de controle são expostas via API REST para orquestradores externos

```
[Simulador gNB]
      │
      │  gRPC streaming  (E2 interface — O-RAN)
      ▼
[gRPC Ingestion Service]
      │
      │  pipeline de métricas
      ▼
[ML Service — RandomForest]
      │
      │  REST API  (A1/O1 interface — O-RAN)
      ▼
[API Gateway — FastAPI]
```

---

## Conexão com O-RAN

| Componente do Projeto | Equivalente O-RAN | Descrição |
|---|---|---|
| `grpc-ingestion` | E2 node (gNB) → near-RT RIC | Coleta métricas do rádio e envia via stream |
| `ml-service` | xApp no near-RT RIC | Detecta anomalias e recomenda ações |
| `api-gateway` | A1 / O1 interface | Expõe decisões para sistemas externos |
| HPA Kubernetes | Elasticidade em telecom | Escala automática sob carga de tráfego |

---

## Stack Tecnológico

| Tecnologia | Papel | Decisão Técnica |
|---|---|---|
| **gRPC + Protocol Buffers** | Ingestão de dados | Streaming nativo, binário 3-10x menor que JSON, padrão em telecom |
| **Python / FastAPI** | API Gateway REST | Alta performance, async-ready, OpenAPI automático |
| **scikit-learn (RandomForest)** | Detecção de anomalias | Inferência <5ms, interpretável, sem GPU |
| **Docker + Compose** | Containerização | Reprodutibilidade e isolamento |
| **Kubernetes + HPA** | Orquestração | Autoscaling baseado em CPU — essencial em ambientes telecom |

---

## Classificação de Anomalias

O modelo detecta 3 estados de rede com base em 4 features:

| Estado | Latência | Throughput | Packet Loss | Ação Recomendada |
|---|---|---|---|---|
| **NORMAL** | 5–30ms | 80–150 Mbps | 0–1% | NENHUMA |
| **CONGESTIONADO** | 80–250ms | 5–40 Mbps | 5–20% | BALANCEAR_CARGA |
| **DEGRADADO** | 200–500ms | 1–15 Mbps | 15–40% | REROUTING_EMERGENCIAL |

---

## Quick Start

### Opção 1 — Notebook (sem Docker, zero configuração)

```bash
pip install jupyter scikit-learn numpy pandas matplotlib seaborn fastapi uvicorn httpx grpcio grpcio-tools
jupyter notebook notebooks/ai_network_optimizer.ipynb
```

Execute as 9 etapas em sequência:

| Etapa | O que faz |
|---|---|
| 0 | Instala dependências (checa o que já existe) |
| 1 | Simulador de métricas de rede (gNB) |
| 2 | Dataset de treino + visualização das features |
| 3 | Treinamento + matriz de confusão + feature importance |
| 4 | Pipeline de inferência em tempo real |
| 5 | Sobe FastAPI na porta 8000 e testa endpoints via httpx |
| 6 | Benchmark de latência e ROC-AUC |
| 7 | Compila `.proto` e demonstra streaming gRPC |
| 8 | Comandos Docker e Kubernetes |
| 9 | Respostas de entrevista baseadas no código real |

### Opção 2 — Docker Compose (ambiente completo local)

```bash
cd infra/
docker-compose up --build
```

| Serviço | URL |
|---|---|
| API REST | http://localhost:8000 |
| Swagger / Docs | http://localhost:8000/docs |
| gRPC | localhost:50051 |

### Opção 3 — Kubernetes (produção)

```bash
# Build das imagens
docker build -t ai-network-optimizer/grpc-ingestion:latest services/grpc-ingestion/
docker build -t ai-network-optimizer/ml-service:latest    services/ml-service/
docker build -t ai-network-optimizer/api-gateway:latest   services/api-gateway/

# Deploy
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/

# Verificar pods e autoscaling
kubectl get pods -n ai-network
kubectl get hpa  -n ai-network

# Acessar API
kubectl port-forward svc/api-gateway-service 8000:80 -n ai-network
```

---

## Endpoints REST

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Info do serviço |
| GET | `/health` | Health check |
| GET | `/network/status` | Status atual de todos os nós (last seen) |
| GET | `/network/alerts` | Somente anomalias detectadas |
| GET | `/network/history?limit=N` | Histórico de inferências |
| GET | `/network/node/{id}` | Status de um nó específico |

Exemplo de resposta `/network/status`:
```json
{
  "overall_status": "CONGESTIONADO",
  "total_nodes": 3,
  "nodes": [
    {
      "node_id": "gNB-001",
      "status": "CONGESTIONADO",
      "confidence": 0.97,
      "action": "BALANCEAR_CARGA",
      "raw_metrics": {
        "latency": 142.5,
        "throughput": 22.3,
        "packet_loss": 11.2,
        "jitter": 38.7
      }
    }
  ]
}
```

---

## Estrutura do Projeto

```
ai-network-optimizer/
├── services/
│   ├── grpc-ingestion/          # Servidor gRPC — simula E2 interface
│   │   ├── app/
│   │   │   ├── server.py        # Servidor gRPC
│   │   │   ├── simulator.py     # Gerador de métricas de rede
│   │   │   └── proto/
│   │   │       └── metrics.proto
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── ml-service/              # Detecção de anomalias com IA
│   │   ├── app/
│   │   │   ├── model.py         # Predict + feature extraction
│   │   │   ├── trainer.py       # Treinamento do RandomForest
│   │   │   └── inference.py     # Worker contínuo de inferência
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── api-gateway/             # REST API — exposição de decisões
│       ├── app/
│       │   ├── main.py          # FastAPI app + startup hooks
│       │   └── routes.py        # Endpoints REST
│       ├── Dockerfile
│       └── requirements.txt
├── infra/
│   ├── docker-compose.yml       # Ambiente local completo
│   └── k8s/
│       ├── namespace.yaml
│       ├── grpc-deployment.yaml
│       ├── ml-deployment.yaml
│       ├── api-deployment.yaml
│       ├── services.yaml
│       └── hpa.yaml             # Autoscaling (2–8 replicas)
├── notebooks/
│   └── ai_network_optimizer.ipynb   # Pipeline completo executável
├── shared/
│   └── schemas/
│       └── metric_schema.py
└── README.md
```

---

## Trade-offs Documentados

**gRPC na ingestão vs REST:**
> gRPC vence em streaming contínuo (essencial em telecom), tipagem forte via proto e eficiência binária. REST vence em universalidade e integração com sistemas legados. Aqui: gRPC onde os dados fluem, REST onde são consumidos.

**RandomForest vs Deep Learning:**
> RF entrega inferência abaixo de 5ms sem GPU, com interpretabilidade via feature importance. LSTM seria superior para padrões temporais complexos, mas adiciona latência e complexidade operacional desnecessária para o escopo deste problema.

**Kubernetes vs Docker Compose:**
> Compose é ideal para desenvolvimento local e demos. K8s é necessário para HA, autoscaling e multi-node — o HPA escala `ml-service` de 2 a 8 réplicas automaticamente sob carga.

---

## Contexto: O-RAN e xApps

O **O-RAN (Open Radio Access Network)** é uma iniciativa para abrir e virtualizar redes móveis, permitindo que componentes de diferentes fornecedores interoperem. O **RIC (RAN Intelligent Controller)** é o componente central que executa aplicações de IA/ML para otimização em tempo real.

- **xApp** → roda no near-RT RIC (horizonte de 10ms–1s), reage a métricas em tempo quase real
- **rApp** → roda no non-RT RIC, análise estratégica de longo prazo
- **E2 interface** → canal entre RIC e os rádios (gNB), onde fluem as métricas
- **A1/O1 interface** → interface norte do RIC, consumida por orquestradores externos

Este projeto simula o ciclo completo: **E2 → near-RT RIC (xApp) → A1**.

---

## Autor

**Yuri Fernando** — [github.com/Yuri-Fernando](https://github.com/Yuri-Fernando)
